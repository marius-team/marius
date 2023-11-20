//
// Created by Jason Mohoney on 2/12/21.
//

#include "nn/model.h"

#ifdef MARIUS_CUDA
    #include <torch/csrc/cuda/nccl.h>
#endif

#include "configuration/constants.h"
#include "data/samplers/negative.h"
#include "nn/decoders/edge/decoder_methods.h"
#include "nn/layers/embedding/embedding.h"
#include "nn/model_helpers.h"
#include "reporting/logger.h"

Model::Model(shared_ptr<GeneralEncoder> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Reporter> reporter,
             LearningTask learning_task, std::vector<shared_ptr<Optimizer>> optimizers) : device_(torch::Device(torch::kCPU)) {
    encoder_ = encoder;
    decoder_ = decoder;
    loss_function_ = loss;
    reporter_ = reporter;
    optimizers_ = optimizers;
//    std::cout<<"1\n";
    if (decoder_ != nullptr) {
        learning_task_ = decoder_->learning_task_;
    } else {
        std::cout<<"set learning task\n";
        learning_task_ = learning_task;
    }
//    std::cout<<"2\n";

    if (reporter_ == nullptr) {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            reporter_ = std::make_shared<LinkPredictionReporter>();
            reporter_->addMetric(std::make_shared<MeanRankMetric>());
            reporter_->addMetric(std::make_shared<MeanReciprocalRankMetric>());
            reporter_->addMetric(std::make_shared<HitskMetric>(1));
            reporter_->addMetric(std::make_shared<HitskMetric>(3));
            reporter_->addMetric(std::make_shared<HitskMetric>(5));
            reporter_->addMetric(std::make_shared<HitskMetric>(10));
            reporter_->addMetric(std::make_shared<HitskMetric>(50));
            reporter_->addMetric(std::make_shared<HitskMetric>(100));
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            reporter_ = std::make_shared<NodeClassificationReporter>();
            reporter_->addMetric(std::make_shared<CategoricalAccuracyMetric>());
        } else {
            throw MariusRuntimeException("Reporter must be specified for this learning task.");
        }
    }

    if (encoder_ != nullptr) {
        register_module("encoder", std::dynamic_pointer_cast<torch::nn::Module>(encoder_));
    }

    if (decoder_ != nullptr) {
        register_module("decoder", std::dynamic_pointer_cast<torch::nn::Module>(decoder_));
    }
}

void Model::clear_grad() {
#pragma omp parallel for
    for (int i = 0; i < optimizers_.size(); i++) {
        optimizers_[i]->clear_grad();
    }
}

void Model::clear_grad_all() {
    for (int i = 0; i < device_models_.size(); i++) {
        device_models_[i]->clear_grad();
    }
}

void Model::step() {
#pragma omp parallel for
    for (int i = 0; i < optimizers_.size(); i++) {
        optimizers_[i]->step();
    }
}

void Model::step_all() {
    for (int i = 0; i < device_models_.size(); i++) {
        device_models_[i]->step();
    }
}

void Model::save(std::string directory) {
    string model_filename = directory + PathConstants::model_file;
    string model_state_filename = directory + PathConstants::model_state_file;
    string model_meta_filename = directory + PathConstants::model_config_file;

    torch::serialize::OutputArchive model_archive;
    torch::serialize::OutputArchive state_archive;

    std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->save(model_archive);

    if (decoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->save(model_archive);
    }

    // Outputs each optimizer as a <K, V> pair, where key is the loop counter and value
    // is the optimizer itself. in Model::load, Optimizer::load is called on each key.
    for (int i = 0; i < optimizers_.size(); i++) {
        torch::serialize::OutputArchive optim_archive;
        optimizers_[i]->save(optim_archive);
        state_archive.write(std::to_string(i), optim_archive);
    }

    model_archive.save_to(model_filename);
    state_archive.save_to(model_state_filename);
}

void Model::load(std::string directory, bool train) {
    string model_filename = directory + PathConstants::model_file;
    string model_state_filename = directory + PathConstants::model_state_file;

    torch::serialize::InputArchive model_archive;
    torch::serialize::InputArchive state_archive;

    model_archive.load_from(model_filename);

    if (train) {
        state_archive.load_from(model_state_filename);
    }

    int optimizer_idx = 0;
    for (auto key : state_archive.keys()) {
        torch::serialize::InputArchive tmp_state_archive;
        state_archive.read(key, tmp_state_archive);
        // optimizers have already been created as part of initModelFromConfig
        optimizers_[optimizer_idx++]->load(tmp_state_archive);
    }

    std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->load(model_archive);

    if (decoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->load(model_archive);
    }
}

void Model::all_reduce() {
    torch::NoGradGuard no_grad;
    int num_gpus = device_models_.size();

//    std::vector<at::cuda::CUDAStream> streams;
//    for (int i = 0; i < stream_ptrs.size(); i++) {
//        streams.emplace_back(*stream_ptrs[i]);
//    }

    for (int i = 0; i < named_parameters().keys().size(); i++) {
        string key = named_parameters().keys()[i];

        std::vector<torch::Tensor> input_gradients(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            if (!device_models_[j]->named_parameters()[key].mutable_grad().defined()) {
                device_models_[j]->named_parameters()[key].mutable_grad() = torch::zeros_like(device_models_[j]->named_parameters()[key]);
            }
            // this line for averaging
            device_models_[j]->named_parameters()[key].mutable_grad() /= (float_t) num_gpus;

            input_gradients[j] = device_models_[j]->named_parameters()[key].mutable_grad();
        }

#ifdef MARIUS_CUDA
        // TODO: want to look at the streams for this?, reduction mean on own
        torch::cuda::nccl::all_reduce(input_gradients, input_gradients, 0);//, streams);
#endif
    }

//    step_all();
//    clear_grad_all();
}

void Model::setup_optimizers(shared_ptr<ModelConfig> model_config) {
    if (model_config->dense_optimizer == nullptr) {
        throw UnexpectedNullPtrException();
    }

    // need to assign named parameters to each optimizer
    torch::OrderedDict<shared_ptr<OptimizerConfig>, torch::OrderedDict<std::string, torch::Tensor>> param_map;

    {
        torch::OrderedDict<std::string, torch::Tensor> empty_dict;
        param_map.insert(model_config->dense_optimizer, empty_dict);
    }

    // get optimizers we need to keep track of for the encoder
    for (auto module_name : encoder_->named_modules().keys()) {
        if (module_name.empty()) {
            continue;
        }
        auto layer = std::dynamic_pointer_cast<Layer>(encoder_->named_modules()[module_name]);
        if (layer->config_->optimizer == nullptr) {
            for (auto param_name : layer->named_parameters().keys()) {
                param_map[model_config->dense_optimizer].insert(module_name + "_" + param_name, layer->named_parameters()[param_name]);
            }
        } else {
            if (!param_map.contains(layer->config_->optimizer)) {
                torch::OrderedDict<std::string, torch::Tensor> empty_dict;
                param_map.insert(layer->config_->optimizer, empty_dict);
            }

            for (auto param_name : layer->named_parameters().keys()) {
                param_map[layer->config_->optimizer].insert(module_name + "_" + param_name, layer->named_parameters()[param_name]);
            }
        }
    }

    for (auto key : std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->named_parameters().keys()) {
        param_map[model_config->dense_optimizer].insert(key, std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->named_parameters()[key]);
    }

    for (auto key : param_map.keys()) {
        switch (key->type) {
            case OptimizerType::SGD: {
                optimizers_.emplace_back(std::make_shared<SGDOptimizer>(param_map[key], key->options->learning_rate));
                break;
            }
            case OptimizerType::ADAGRAD: {
                optimizers_.emplace_back(std::make_shared<AdagradOptimizer>(param_map[key], std::dynamic_pointer_cast<AdagradOptions>(key->options)));
                break;
            }
            case OptimizerType::ADAM: {
                optimizers_.emplace_back(std::make_shared<AdamOptimizer>(param_map[key], std::dynamic_pointer_cast<AdamOptions>(key->options)));
                break;
            }
            default:
                throw std::invalid_argument("Unrecognized optimizer type");
        }
    }
}

int64_t Model::get_base_embedding_dim() {
//    int max_offset = 0;
//    int size = 0;
//
//    for (auto stage : encoder_->layers_) {
//        for (auto layer : stage) {
//            if (layer->config_->type == LayerType::EMBEDDING) {
//                int offset = std::dynamic_pointer_cast<EmbeddingLayer>(layer)->offset_;
//
//                if (size == 0) {
//                    size = layer->config_->output_dim;
//                }
//
//                if (offset > max_offset) {
//                    max_offset = offset;
//                    size = layer->config_->output_dim;
//                }
//            }
//        }
//    }
//
//    return max_offset + size;

    int size = 0;

    if (model_config_->encoder != nullptr) {
        for (auto stage_config : model_config_->encoder->layers) {
            for (auto layer_config : stage_config) {
                if (layer_config->type == LayerType::EMBEDDING) {

                    if (size == 0) {
                        size = layer_config->output_dim;
                    }
                }
            }
        }
    }

    std::cout<<"Embedding dim: "<<size<<"\n";
    return size;
}

bool Model::has_embeddings() { return has_embeddings_; }

bool Model::has_partition_embeddings() { return has_partition_embeddings_; }

torch::Tensor Model::forward_nc(at::optional<torch::Tensor> node_embeddings, at::optional<torch::Tensor> node_features, DENSEGraph dense_graph, bool train) {
    torch::Tensor encoded_nodes = encoder_->forward(node_embeddings, node_features, dense_graph, train);
    torch::Tensor y_pred = std::dynamic_pointer_cast<NodeDecoder>(decoder_)->forward(encoded_nodes);
    return y_pred;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Model::forward_lp(shared_ptr<Batch> batch, bool train) {
    torch::Tensor encoded_nodes = encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, train);

    // call proper decoder
    torch::Tensor pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor inv_neg_scores;

    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);

    if (edge_decoder->decoder_method_ == EdgeDecoderMethod::ONLY_POS) {
        std::tie(pos_scores, inv_pos_scores) = only_pos_forward(edge_decoder, batch->edges_, encoded_nodes);
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::POS_AND_NEG) {
        throw MariusRuntimeException("Decoder method currently unsupported.");
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) = neg_and_pos_forward(edge_decoder, batch->edges_, batch->neg_edges_, encoded_nodes);
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_NODE) {
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
            node_corrupt_forward(edge_decoder, batch->edges_, encoded_nodes, batch->dst_neg_indices_mapping_, batch->src_neg_indices_mapping_);
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_REL) {
        throw MariusRuntimeException("Decoder method currently unsupported.");
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
            rel_corrupt_forward(edge_decoder, batch->edges_, encoded_nodes, batch->rel_neg_indices_);
    } else {
        throw MariusRuntimeException("Unsupported encoder method");
    }

    if (neg_scores.defined()) {
        neg_scores = apply_score_filter(neg_scores, batch->dst_neg_filter_);
    }

    if (inv_neg_scores.defined()) {
        inv_neg_scores = apply_score_filter(inv_neg_scores, batch->src_neg_filter_);
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

void Model::train_batch(shared_ptr<Batch> batch, bool call_step) {
//    torch::cuda::nccl::ncclComm_t comms[2];
//    int devs[2] = {0, 1};
//    torch::cuda::nccl::ncclCommInitAll(comms, 2, devs);
//    torch::cuda::nccl::ncclUniqueId Id;
//    torch::cuda::nccl::get_unique_id(Id);
//
//    torch::cuda::nccl::ncclUniqueId Id1;
//    torch::cuda::nccl::get_unique_id(Id1);
//
//    torch::cuda::nccl::ncclComm_t comm1 = torch::cuda::nccl::comm_init_rank(2, Id, 0);
//    torch::cuda::nccl::ncclComm_t comm2 = torch::cuda::nccl::comm_init_rank(2, Id1, 1);

//    Timer t = new Timer(false);
//    t.start();

//    batch->node_features_ = batch->node_features_.to(torch::Device("cuda:1"));
//    batch->node_features_ = batch->node_features_.to(torch::Device("cuda:0"));

//    torch::Tensor test = torch::zeros_like(batch->node_features_).to(torch::Device("cuda:1"));

//    std::cout<<torch::cuda::nccl::detail::get_communicators({batch->node_features_})[0]<<"\n";
//    std::cout<<at::cuda::getCurrentCUDAStream(0)<<"\n";
//    torch::cuda::nccl::send(batch->node_features_, torch::cuda::nccl::NcclCommList, at::cuda::getCurrentCUDAStream(0), 1);
//    torch::cuda::nccl::recv(test, torch::cuda::nccl::detail::get_communicators({test})[0], at::cuda::getCurrentCUDAStream(1), 0);

//    std::cout<<test.flatten().narrow(0, 0, 2)<<"\n";

//    t.stop();
//    std::cout<<"blah: "<<t.getDuration()<<"\n\n";





    if (batch->sub_batches_.size() > 0) {
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < batch->sub_batches_.size(); i++) {
                device_models_[i]->clear_grad();
                device_models_[i]->train_batch(batch->sub_batches_[i], false);
            }

            #pragma omp single
            {
                all_reduce();
                distGradSync();
            }

            #pragma omp for
            for (int i = 0; i < batch->sub_batches_.size(); i++) {
                device_models_[i]->step();
            }

            #pragma omp single
            {
                distModelSync();
            }
        }
        return;
    }

    // single GPU
    if (call_step) {
        clear_grad();
    }

    if (batch->node_embeddings_.defined()) {
        batch->node_embeddings_.requires_grad_();
    }

    torch::Tensor loss;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto all_scores = forward_lp(batch, true);

        torch::Tensor pos_scores = std::get<0>(all_scores);
        torch::Tensor neg_scores = std::get<1>(all_scores);
        torch::Tensor inv_pos_scores = std::get<2>(all_scores);
        torch::Tensor inv_neg_scores = std::get<3>(all_scores);

        if (inv_neg_scores.defined()) {
            torch::Tensor rhs_loss = loss_function_->operator()(pos_scores, neg_scores, true);
            torch::Tensor lhs_loss = loss_function_->operator()(inv_pos_scores, inv_neg_scores, true);
            loss = lhs_loss + rhs_loss;
        } else {
            loss = (*loss_function_)(pos_scores, neg_scores, true);
        }

    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        torch::Tensor y_pred = forward_nc(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, true);
        loss = (*loss_function_)(y_pred, batch->node_labels_.to(torch::kInt64), false);
    } else {
        throw MariusRuntimeException("Unsupported learning task for training");
    }

    loss.backward();

    batch->loss_ = (double) loss.item<float>();

    if (call_step) {
        distGradSync();
        step();
        distModelSync();
    }

    if (batch->node_embeddings_.defined()) {
        batch->accumulateGradients(sparse_lr_);
    }

//    std::cout<<"train_batch\n";
//    std::cout<<named_parameters()[named_parameters().keys()[0]].flatten().narrow(0, 0, 10)<<"\n";
}

void Model::evaluate_batch(shared_ptr<Batch> batch, bool add_result_to_batch) {
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto all_scores = forward_lp(batch, true);
        torch::Tensor pos_scores = std::get<0>(all_scores);
        torch::Tensor neg_scores = std::get<1>(all_scores);
        torch::Tensor inv_pos_scores = std::get<2>(all_scores);
        torch::Tensor inv_neg_scores = std::get<3>(all_scores);

        if (!add_result_to_batch) {
            if (neg_scores.defined()) {
                std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(pos_scores, neg_scores);
            }

            if (inv_neg_scores.defined()) {
                std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(inv_pos_scores, inv_neg_scores);
            }
        } else {
            batch->pos_scores_ = pos_scores;
            batch->neg_scores_ = neg_scores;
            batch->inv_pos_scores_ = inv_pos_scores;
            batch->inv_neg_scores_ = inv_neg_scores;
        }
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        torch::Tensor y_pred = forward_nc(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, true);
        torch::Tensor labels = batch->node_labels_;

        if (!add_result_to_batch) {
            std::dynamic_pointer_cast<NodeClassificationReporter>(reporter_)->addResult(labels, y_pred);
        } else {
            batch->y_pred_ = y_pred;
        }
    } else {
        throw MariusRuntimeException("Unsupported learning task for evaluation");
    }
}

void Model::createDeviceModels(std::vector<torch::Device> devices) {
    int i = 0;
    for (auto device : devices) {
        SPDLOG_INFO("Broadcast to GPU {}", device.index());
        if (device != device_) {
            shared_ptr<GeneralEncoder> encoder = encoder_clone_helper(encoder_, device);
            shared_ptr<Decoder> decoder = decoder_clone_helper(decoder_, device);
            device_models_[i] = std::make_shared<Model>(encoder, decoder, loss_function_, reporter_);
            device_models_[i]->device_ = device;

            // encoder_clone_helper triggers a reset() call we need to undo
            // copy to correct device, not sure why encoder_clone_helper doesn't work
            torch::NoGradGuard no_grad;

            if (encoder != nullptr) {
                for (int i = 0; i < encoder->named_parameters().values().size(); i++) {
                    encoder->named_parameters().values()[i].copy_(device_models_[0]->encoder_->named_parameters().values()[i]);
                }
                encoder->to(device);
            }
            if (decoder != nullptr) {
                for (int i = 0; i < std::dynamic_pointer_cast<torch::nn::Module>(decoder)->named_parameters().values().size(); i++) {
                    std::dynamic_pointer_cast<torch::nn::Module>(decoder)->named_parameters().values()[i].copy_(
                            std::dynamic_pointer_cast<torch::nn::Module>(device_models_[0]->decoder_)->named_parameters().values()[i]);
                }
                std::dynamic_pointer_cast<torch::nn::Module>(decoder)->to(device);
            }

            device_models_[i]->setup_optimizers(model_config_); // TODO: change this to broadcast optimizers to we can actually use this as a broadcast function?
            device_models_[i]->sparse_lr_ = sparse_lr_;
        } else {
            device_models_[i] = std::dynamic_pointer_cast<Model>(shared_from_this());
        }
        i++;
    }
}

void Model::distGradSync() {
    if (pg_gloo_ == nullptr or (compute_pg_ == nullptr and compute_pg_nccl_ == nullptr))
        return;
    if (dist_config_->model_sync != DistributedModelSync::SYNC_GRADS)
        return;

    bool success = false;
    while (!success) {
        try {
            pg_lock_->lock();

            torch::NoGradGuard no_grad;
            int num_gpus = device_models_.size();

            shared_ptr<c10d::Backend> pg = compute_pg_;
            if (pg == nullptr) {
                pg = compute_pg_nccl_;
            }
//            if (global_pg) {
//                pg = total_compute_pg_;
//            }

//            #pragma omp parallel for
            for (int i = 0; i < named_parameters().keys().size(); i++) {
                string key = named_parameters().keys()[i];

                std::vector<torch::Tensor> transfer_vec(1);
                if (!named_parameters()[key].mutable_grad().defined()) {
                    named_parameters()[key].mutable_grad() = torch::zeros_like(named_parameters()[key]);
                }
                transfer_vec[0] = named_parameters()[key].mutable_grad();

                // all reduce
                auto options = c10d::AllreduceOptions();
                options.reduceOp = c10d::ReduceOp::SUM;
                auto work = pg->allreduce(transfer_vec, options);
                if (!work->wait()) {
                    throw work->exception();
                }
                // manually perform average
                named_parameters()[key].mutable_grad() /= (float_t) pg->getSize();

//                std::cout<<named_parameters()[key].mutable_grad().flatten().narrow(0, 0, 5)<<"\n";

                // local device model broadcast
                std::vector<torch::Tensor> tensors(num_gpus);
                for (int j = 0; j < num_gpus; j++) {
                    tensors[j] = named_parameters()[key].mutable_grad();
                }

                #ifdef MARIUS_CUDA
                    torch::cuda::nccl::broadcast(tensors); //, streams);  // TODO: want to look at the streams for this?
                #endif

//                for (int j = 0; j < num_gpus; j++) {
//                    std::cout<<"j2 "<<j<<" :"<<device_models_[j]->named_parameters()[key].flatten().narrow(0, 0, 10)<<"\n";
//                }
            }

            pg_lock_->unlock();

            success = true;
        } catch (...) {
            pg_lock_->unlock();
            std::cout<<"Caught ERROR with distGradSync\n";
        }
    }

}

void Model::distModelSync(bool global_pg, bool bypass_check, bool all_reduce, bool optimizers, int from_worker) {
    if (pg_gloo_ == nullptr or (compute_pg_ == nullptr and compute_pg_nccl_ == nullptr and total_compute_pg_ == nullptr))
        return;
    if (dist_config_->model_sync != DistributedModelSync::SYNC_MODEL and !bypass_check)
        return;

    std::cout<<"DIST_MODEL_SYNC\n";
    std::cout<<all_reduce<<" "<<from_worker<<"\n";

    // NOTE that this assumes that the device models/optimizers on each machine are already the same, i.e. only the model on GPU 0
    // participates in the global sync, then this sync is broadcast to the other gpus
//    std::cout<<"dist model sync\n";

    // TODO: perf: maybe this (and other similar functions)
    //  should use some sort of all reduce coalesced, and or nccl backend for compute pg? omp doesn't seem to work

    // TODO: why is the acc different?

    bool success = false;
    while (!success) {
        try {
            pg_lock_->lock();

//            auto work = compute_pg_->allreduce(vec);
//            if (!work->wait(std::chrono::milliseconds(1000))) {
//             // std::cout << "err\n";
//                throw work->exception();
//            }
//            std::cout<<"try distModelSync\n";



            torch::NoGradGuard no_grad;
            int num_gpus = device_models_.size();

            shared_ptr<c10d::Backend> pg = compute_pg_;
            if (pg == nullptr) {
                std::cout<<"nccl\n";
                pg = compute_pg_nccl_;
            }
            if (global_pg) {
                std::cout<<"global\n";
                pg = total_compute_pg_;
            }

//            std::vector<torch::Tensor> transfer_vec(named_parameters().keys().size());
//            #pragma omp parallel for
            for (int i = 0; i < named_parameters().keys().size(); i++) {
                string key = named_parameters().keys()[i];
//                transfer_vec[i] = named_parameters()[key].data();

//                std::cout<<"named parameters: "<<i<<"\n";
//                std::cout<<"key: "<<key<<"\n";
//                std::cout<<"transfer_vec[i]: "<<transfer_vec[i].sizes()<<"\n";
//                std::cout<<"transfer_vec[i]: "<<transfer_vec[i].device()<<"\n";


//                named_parameters()[key] = 1*named_parameters()[key];
                std::vector<torch::Tensor> transfer_vec(1);
                transfer_vec[0] = named_parameters()[key].data();

//                pg_lock_->lock();

                if (!all_reduce) {
                    auto options = c10d::BroadcastOptions();
                    options.rootRank = from_worker;
                    auto work = pg->broadcast(transfer_vec, options);
                    if (!work->wait()) {
                        throw work->exception();
                    }

//                    named_parameters()[key].copy_(transfer_vec[0]);
                } else {
                    auto options = c10d::AllreduceOptions();
                    options.reduceOp = c10d::ReduceOp::SUM;
                    auto work = pg->allreduce(transfer_vec, options);
                    if (!work->wait()) {
                        throw work->exception();
                    }
                    // manually perform average
                    named_parameters()[key].data() /= (float_t) pg->getSize();
                }

//                pg_lock_->unlock();


//                std::cout<<"local broadcast: "<<i<<"\n";
                // local device model broadcast
                std::vector<torch::Tensor> tensors(num_gpus);
                for (int j = 0; j < num_gpus; j++) {
                    tensors[j] = device_models_[j]->named_parameters()[key].data();
//                    std::cout<<"j "<<j<<" :"<<device_models_[j]->named_parameters()[key].flatten().narrow(0, 0, 10)<<"\n";
                }

                #ifdef MARIUS_CUDA
                torch::cuda::nccl::broadcast(tensors); //, streams);  // TODO: want to look at the streams for this?
                #endif

//                for (int j = 0; j < num_gpus; j++) {
//                    std::cout<<"j2 "<<j<<" :"<<device_models_[j]->named_parameters()[key].flatten().narrow(0, 0, 10)<<"\n";
//                }

//                std::cout<<"js\n";
//                std::cout<<named_parameters()[key].flatten().narrow(0, 0, 10)<<"\n";
            }

            if (optimizers) {
                for (int i = 0; i < optimizers_.size(); i++) {
//                    std::cout<<"optimizer_: "<<i<<"\n";

//                    #pragma omp parallel for
                    for (int j = 0; j < optimizers_[i]->state_dict_.keys().size(); j++) {
                        string key = optimizers_[i]->state_dict_.keys()[j];
//                        std::cout<<"state_dict_: "<<key<<"\n";

                        for (int k = 0; k < optimizers_[i]->state_dict_[key].size(); k++) {
                            string param_key = optimizers_[i]->state_dict_[key].keys()[k];
//                            std::cout<<"param_key: "<<param_key<<"\n";

                            std::vector<torch::Tensor> transfer_vec(1);
                            transfer_vec[0] = optimizers_[i]->state_dict_[key][param_key].data();
//                            std::cout<<transfer_vec[0].sizes()<<"\n";


//                            pg_lock_->lock();

                            if (!all_reduce) {
                                auto options = c10d::BroadcastOptions();
                                options.rootRank = from_worker;
                                auto work = pg->broadcast(transfer_vec, options);
                                if (!work->wait()) {
                                    throw work->exception();
                                }
                            } else {
                                auto options = c10d::AllreduceOptions();
                                options.reduceOp = c10d::ReduceOp::SUM;
                                auto work = pg->allreduce(transfer_vec, options);
                                if (!work->wait()) {
                                    throw work->exception();
                                }
                                // manually perform average
                                optimizers_[i]->state_dict_[key][param_key].data() /= (float_t) pg->getSize();
                            }

//                            pg_lock_->unlock();


                            // local device model broadcast
                            std::vector<torch::Tensor> tensors(num_gpus);
                            for (int l = 0; l < num_gpus; l++) {
                                tensors[l] = device_models_[l]->optimizers_[i]->state_dict_[key][param_key].data();
//                                std::cout<<"j "<<j<<" :"<<device_models_[j]->named_parameters()[key].flatten(0, 1).narrow(0, 0, 10)<<"\n";
                            }

                            #ifdef MARIUS_CUDA
                            torch::cuda::nccl::broadcast(tensors); //, streams);  // TODO: want to look at the streams for this?
                            #endif

                        }
                    }
                }
            }

            pg_lock_->unlock();

            success = true;
            std::cout << "success\n";
        } catch (...) {
            pg_lock_->unlock();
            std::cout<<"Caught ERROR with distModelSync\n\n";
        }
    }

//    std::cout<<"done distModelSync\n";

}

void Model::createComputePG(vector<vector<int>> feeders, vector<int> global_to_local, vector<int> local_to_global) {
    std::cout<<"createComputePG\n";

    // only compute workers call this function

    int my_rank;
    int participating_count = 0;
    int lead_compute_worker = -1;
    bool is_participating = false;

    vector<int> remaining_compute_workers;
    for (int i = 0; i < feeders.size(); i++) {
        if (feeders[i].size() > 0) {
            lead_compute_worker = i;
            remaining_compute_workers.push_back(i);

            if (local_to_global[i] == pg_gloo_->pg->getRank()) {
                is_participating = true;
                my_rank = participating_count;
            }

            participating_count++;
        }
    }

    if (!is_participating)
        return;

    lead_compute_worker = local_to_global[lead_compute_worker];

    torch::Tensor addr = torch::from_blob((void *) pg_gloo_->address.c_str(), {(long) pg_gloo_->address.length()}, torch::kInt8);
    addr = addr.clone();

    std::cout<<"a\n";

    if (pg_gloo_->pg->getRank() == lead_compute_worker) {
        std::vector<torch::Tensor> transfer_vec;
        transfer_vec.push_back(addr);
        for (auto compute_worker_id : remaining_compute_workers) {
            compute_worker_id = local_to_global[compute_worker_id];
            if (compute_worker_id == lead_compute_worker)
                continue;
            auto work = pg_gloo_->pg->send(transfer_vec, compute_worker_id, 1); //TODO: change this to a broadcast?
            if (!work->wait()) {
                throw work->exception();
            }
        }
    } else {
        std::vector<torch::Tensor> transfer_vec;
        transfer_vec.push_back(addr);
        auto work = pg_gloo_->pg->recv(transfer_vec, lead_compute_worker, 1);
        if (!work->wait()) {
            throw work->exception();
        }
    }

    std::cout<<"b\n";

    string coord_addr;
    coord_addr.assign((char *)addr.data_ptr(), (char *)addr.data_ptr()+addr.size(0));

//    std::cout<<coord_addr<<"\n";
//    std::cout<<pg_gloo_->address<<"\n";
//    std::cout<<num_compute_workers<<"\n";
//    std::cout<<(pg_gloo_->pg->getRank()==lead_compute_worker)<<"\n";
//    std::cout<<rank<<"\n";

    pg_lock_->lock();
    std::cout<<"c\n";
    compute_pg_ = nullptr;
    compute_pg_nccl_ = nullptr; // TODO: clearing this seems slow, especially when using nccl
    std::cout<<"cd\n";
    auto store = c10::make_intrusive<c10d::TCPStore>(coord_addr, 7655, participating_count,
                                                     pg_gloo_->pg->getRank()==lead_compute_worker);

    std::cout<<"d\n";

    if (false) {
        auto options = c10d::ProcessGroupGloo::Options::create();
        options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname(pg_gloo_->address));
//        options.timeout = std::chrono::milliseconds(1000);
//        options.threads =

        compute_pg_ = std::make_shared<c10d::ProcessGroupGloo>(store, my_rank, participating_count, options);
    } else {
        auto options = c10d::ProcessGroupNCCL::Options::create();
//        options->devices.push_back(c10d::ProcessGroupNCCL::createDeviceForHostname(pg_gloo_->address));
//        options.timeout = std::chrono::milliseconds(1000);
//        options.threads =

        std::cout<<my_rank<<" "<<participating_count<<"\n";
        if (participating_count > 1)
            compute_pg_nccl_ = std::make_shared<c10d::ProcessGroupNCCL>(store, my_rank, participating_count, options);
    }

    if (total_compute_pg_ == nullptr) {
        auto store = c10::make_intrusive<c10d::TCPStore>(coord_addr, 7656, participating_count,
                                                         pg_gloo_->pg->getRank()==lead_compute_worker);

        auto options = c10d::ProcessGroupGloo::Options::create();
        options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname(pg_gloo_->address));
//    options.timeout = std::chrono::milliseconds(1000);
//    options.threads =

        total_compute_pg_ = std::make_shared<c10d::ProcessGroupGloo>(store, my_rank, participating_count, options);
    }

    pg_lock_->unlock();

    std::cout<<"done createComputePG\n";
}

void Model::distPrepareForTraining(bool eval) {
    if (pg_gloo_ == nullptr) {
        return;
    }

    already_notified_ = false;

    std::cout<<"distPrepareForTraining\n";

    // set batch_worker_, compute_worker_, children, parents, num_batch_, num_compute_ here based on config,
    //  or maybe set these in model if they are needed there and the pipeline
    //  initialize compute worker group here (to all compute workers) and initialize compute worker feeders etc.

    epoch_complete_ = false;
//    last_compute_worker_ = -1;

    int ii = 0;
    int jj = 0;

    vector<int> compute_worker_to_global; // size compute workers, holds global ids
    vector<int> global_to_compute_worker; // size all workers, holds compute id or -1

    int num_compute_workers = 0;

    for (auto worker_config : dist_config_->workers) {
        bool compute_worker = false;
        if (worker_config->type == WorkerType::BATCH) {
            if (ii == pg_gloo_->pg->getRank()) {
                batch_worker_ = true;
            }
            if (std::dynamic_pointer_cast<BatchWorkerOptions>(worker_config->options)->also_compute == true) {
                compute_worker = true;
            }
        } else if (worker_config->type == WorkerType::COMPUTE) {
            compute_worker = true;
        }

        if (compute_worker) {
            global_to_compute_worker.push_back(jj);
            compute_worker_to_global.push_back(ii);
            num_compute_workers++;

            if (ii == pg_gloo_->pg->getRank()) {
                compute_worker_ = true;
            }

            last_compute_worker_ = jj;

            jj++;
        } else {
            global_to_compute_worker.push_back(-1);
        }
        ii++;
    }

    compute_workers_ = compute_worker_to_global;
    all_workers_ = global_to_compute_worker;


    // everyone tracks the feeders (batch construction workers included) for now
    ii = 0;
    vector<vector<int>> feeders(num_compute_workers);
    for (auto worker_config : dist_config_->workers) {
        if (worker_config->type == WorkerType::BATCH) {
            vector<int> children = std::dynamic_pointer_cast<BatchWorkerOptions>(worker_config->options)->children;
            for (auto c : children) {
                feeders[global_to_compute_worker[c]].push_back(ii);
            }
//            if (ii == pg_gloo_->pg->getRank()) {
//                children_ = children;
//            }
        }
        ii++;
    }

//    std::cout<<"feeders:\n";
//    for (int i = 0; i < feeders.size(); i++) {
//        std::cout<<i<<": "<<feeders[i]<<"\n";
//    }

    feeders_ = feeders;

//    if (compute_worker_ and !eval) {
//        createComputePG(feeders, global_to_compute_worker, compute_worker_to_global);
//    }

    std::thread(&Model::distListenForComplete, this, eval).detach();


    if (first_epoch_ and !eval) {
        // set models equal to model on compute worker zero, only need to do this at the very beginning
        // optimizers should be initialized the same everywhere

//        std::cout<<"size:"<<named_parameters()[named_parameters().keys()[0]].flatten(0, 1).narrow(0, 0, 10)<<"\n";
        distModelSync(true, true, false, false, 0);
//        std::cout<<"size:"<<named_parameters()[named_parameters().keys()[0]].flatten(0, 1).narrow(0, 0, 10)<<"\n";

        first_epoch_ = false;
    }

    std::cout<<"done distPrepareForTraining\n";


    // TODO: batch construction (i.e., everybody) waits for sync (with some sort of barrier)?


    auto work = pg_gloo_->pg->barrier(); // TBD on if this actually works when we have batch construction workers or not
////    while (!work->isCompleted()) { std::cout<<"barrier waiting\n"; }
    if (!work->wait()) {
        throw work->exception();
    }

    std::cout<<"distPrepareForTraining barrier complete\n";


//    exit(0);
}

void Model::updateFeeders(int x, bool eval) {
    std::cout<<"update feeders: "<<x<<"\n";
    update_feeders_lock_->lock();

    if (compute_workers_[0] == pg_gloo_->pg->getRank()) {
        for (int i = 0; i < pg_gloo_->pg->getSize(); i++) {
            if (i == compute_workers_[0]) {
                continue;
            }

            std::cout<<pg_gloo_->pg->getRank()<<" sending "<<x<<" to "<<i<<"\n";

//            bool success = false;

//            while (!success) {
//                try {
            std::vector<torch::Tensor> vec;
            torch::Tensor x_tens = torch::zeros({1}, torch::kInt32) + x;
            vec.push_back(x_tens);
            auto work = pg_gloo_->pg->send(vec, i, 2 + eval);
            if (!work->wait()) {                        //std::chrono::milliseconds(1000)
                throw work->exception();
            }
//                    success = true;
//                } catch (...) {
//                    std::cout<<"Caught ERROR with update feeders\n\n";
//                }
//            }

            std::cout<<"done sending "<<x<<" to "<<i<<"\n";
        }
    }

    for (int i = 0; i < feeders_.size(); i++) {
//        std::cout<<"start\n";
        vector<int>::iterator it = std::find(feeders_[i].begin(), feeders_[i].end(), x);
        if (it != feeders_[i].end()) {
            feeders_[i].erase(it);
        }
    }

    bool all_empty = true;

//    std::cout<<"feeders:\n";
    for (int i = 0; i < feeders_.size(); i++) {
//        std::cout<<i<<": "<<feeders_[i]<<"\n";
        if (feeders_[i].size() != 0) {
            all_empty = false;
            last_compute_worker_ = i;
        }
    }

    if (all_empty) {
//        std::cout<<"epoch should complete\n";
        epoch_complete_ = true;
    }

//    if (compute_worker_ and !eval and !epoch_complete_)
//        createComputePG(feeders_, all_workers_, compute_workers_);

    update_feeders_lock_->unlock();
    std::cout<<"done update feeders:"<<x<<"\n";
}

void Model::distListenForComplete(bool eval) {
    while (!epoch_complete_) {
        std::cout<<"distListenForComplete"<<"\n";

//    auto options = c10d::BroadcastOptions::create();
//    options.rootRank =

        std::vector<torch::Tensor> vec;
        torch::Tensor x = torch::zeros({1}, torch::kInt32) - 1;
        vec.push_back(x);

//        std::cout<<"x: "<<x<<"\n";

        auto work = pg_gloo_->pg->recvAnysource(vec, 2 + eval);
//        std::cout<<"compute_worker[0]: "<< compute_workers_[0]<<"\n";
//        auto work = pg_gloo_->pg->recv(vec, compute_workers_[0], 2 + eval);
//        std::cout<<"distListenForComplete waiting\n";
        if (!work->wait()) {
            throw work->exception();
        }

        updateFeeders(x.item<int>(), eval);

        std::cout<<"done distListenForComplete"<<"\n";
    }
//    std::cout<<"done distListenForComplete"<<"\n";
//    auto work = pg_gloo_->pg->barrier();
//    if (!work->wait()) {
//        throw work->exception();
//    }
    std::cout<<"done done distListenForComplete"<<"\n";
}

//void Model::distModelAllReduce() {
//    torch::NoGradGuard no_grad;
//    int num_gpus = device_models_.size();
//
////    std::vector<at::cuda::CUDAStream> streams;
////    for (int i = 0; i < stream_ptrs.size(); i++) {
////        streams.emplace_back(*stream_ptrs[i]);
////    }
//
//    for (int i = 0; i < named_parameters().keys().size(); i++) {
//        string key = named_parameters().keys()[i];
//
//        std::vector<torch::Tensor> input_gradients(num_gpus);
//        for (int j = 0; j < num_gpus; j++) {
//            if (!device_models_[j]->named_parameters()[key].mutable_grad().defined()) {
//                device_models_[j]->named_parameters()[key].mutable_grad() = torch::zeros_like(device_models_[j]->named_parameters()[key]);
//            }
//            // this line for averaging
//            device_models_[j]->named_parameters()[key].mutable_grad() /= (float_t) num_gpus;
//
//            input_gradients[j] = device_models_[j]->named_parameters()[key].mutable_grad();
//        }
//
//        #ifdef MARIUS_CUDA
//        // want to look at the streams for this?, reduction mean on own
//        torch::cuda::nccl::all_reduce(input_gradients, input_gradients, 0);//, streams);
//        #endif
//    }
//
//
//    // for the optimizer, need to iterate through state_dict, for each key (parameter), need to iterate through "states" e.g., sum for adagrad and then
//    // sync on the tensor
//
////    step_all();
////    clear_grad_all();
//}

//void Model::distModelAverage() {
//    std::cout<<"distModelAvg\n";

    // only called on compute workers

//    auto options = c10d::AllreduceOptions();
//    options.timeout = std::chrono::milliseconds(10000);

//    std::vector<torch::Tensor> vec;
//    torch::Tensor x = torch::randint(10, {5});
//    x = x.to(torch::Device(torch::kCUDA, 0));
//    vec.push_back(x);

//    bool success = false;
//    while (!success) {
//        try {
//            pg_lock_->lock();
//
//            auto work = compute_pg_->allreduce(vec);
//            if (!work->wait(std::chrono::milliseconds(1000))) {
////            std::cout << "err\n";
//                throw work->exception();
//            }
//            pg_lock_->unlock();
//
//            success = true;
////            std::cout << "success\n";
//        } catch (...) {
//            pg_lock_->unlock();
////            std::cout<<"caught timeout\n";
//        }
//    }

//}

void Model::distNotifyCompleteAndWait(bool eval, bool wait) {
    if (pg_gloo_ == nullptr) {
        return;
    }

    std::cout<<"distNotifyCompleteAndWait\n";

    // called on everything

    if (!already_notified_) {
        already_notified_ = true;

        // if batch construction worker, notify all of completion
        if (batch_worker_) {
            torch::Tensor x = torch::zeros({1}, torch::kInt32) + pg_gloo_->pg->getRank();
            std::vector<torch::Tensor> transfer_vec;
            transfer_vec.push_back(x);

            auto compute_worker_id = compute_workers_[0];
    //        for (auto compute_worker_id : compute_workers_) {
    //            std::cout<<compute_worker_id<<"\n";
            if (compute_worker_id == pg_gloo_->pg->getRank()) {
    //            std::cout<<"direct update feeders\n";
                updateFeeders(compute_worker_id, eval); //TODO: this can actually cause update feeders to be called in parallel with the thread, we should have a lock
    //            continue;
            } else {
                std::cout<<pg_gloo_->pg->getRank()<<" direct sending "<<x.item<int>()<<" to "<<compute_worker_id<<"\n";

                auto work = pg_gloo_->pg->send(transfer_vec, compute_worker_id, 2 + eval);
                if (!work->wait()) {
                    throw work->exception();
                }
                std::cout<<"done direct sending\n";
            }

    //        }

    //        if (!compute_worker_) {
    //            std::cout<<"distNotifyCompleteAndWait barrier\n";
    //            auto work = pg_gloo_->pg->barrier();
    //            if (!work->wait()) {
    //                throw work->exception();
    //            }
    //            epoch_complete_ = true;
    //        }
        } else {

        }
    }

    if (!wait) {
        return;
    }


    std::cout<<"distNotifyCompleteAndWait barrier\n";
    while (!epoch_complete_) {}

    std::cout<<"done waiting\n";

//    pg_gloo_->pg->barrier();
//    auto work = pg_gloo_->pg->barrier(); // TBD on if this actually works when we have batch construction workers or not
////    while (!work->isCompleted()) { std::cout<<"barrier waiting\n"; }
//    if (!work->wait()) {
//        throw work->exception();
//    }

//    std::vector<torch::Tensor> vec;
//    torch::Tensor x = torch::randint(10, {5});
//    vec.push_back(x);
//    auto work = pg_gloo_->pg->allreduce(vec);
//    if (!work->wait()) {
//        throw work->exception();
//    }

//    epoch_complete_ = true;

//    auto work = pg_gloo_->pg->barrier(); // TBD on if this actually works when we have batch construction workers or not
////    while (!work->isCompleted()) { std::cout<<"barrier waiting\n"; }
//    if (!work->wait()) {
//        throw work->exception();
//    }

//    std::cout<<"done\n";
//    exit(0);

    std::cout<<"last: "<<last_compute_worker_<<"\n";
    if (!eval)
        distModelSync(true, true, false, true, last_compute_worker_);

    std::cout<<"1\n";
    pg_lock_->lock();
    std::cout<<"2\n";
    compute_pg_ = nullptr; // TODO: clearing these pointers also seems to take a while, probably the compute pg
    compute_pg_nccl_ = nullptr;
    total_compute_pg_ = nullptr;
    pg_lock_->unlock();

//    auto work = pg_gloo_->pg->barrier();
//    if (!work->wait()) {
//        throw work->exception();
//    }

    std::cout<<"done distNotifyCompleteAndWait\n";

    // TODO: batch construction (i.e., everybody) waits for sync (with some sort of barrier)?
}

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, int num_partitions, bool train,
                                      bool compute_worker) {
    // TODO: we don't need to create most of this stuff if we are just a batch construction worker
    //  basically just need to know the learning task, device_, and whether or not there are partition embeddings,
    //  dataloader should also hold bc workers children or something for the pipeline
    //  also, dataloader doesn't need to be null for a compute worker (but can be basically empty), since it needs the streams and such

    shared_ptr<GeneralEncoder> encoder = nullptr;
    shared_ptr<Decoder> decoder = nullptr;
    shared_ptr<LossFunction> loss = nullptr;
    shared_ptr<Model> model;

    if (model_config->encoder == nullptr) {
        throw UnexpectedNullPtrException("Encoder config undefined");
    }

    if (model_config->decoder == nullptr) {
        throw UnexpectedNullPtrException("Decoder config undefined");
    }

    if (model_config->loss == nullptr) {
        throw UnexpectedNullPtrException("Loss config undefined");
    }

    bool has_embeddings = false;
    bool has_partition_embeddings = false;
    if (model_config->encoder != nullptr) {
        for (auto stage_config : model_config->encoder->layers) {
            for (auto layer_config : stage_config) {
                if (layer_config->type == LayerType::EMBEDDING) {
                    has_embeddings = true;
                } else if (layer_config->type == LayerType::PARTITION_EMBEDDING) {
                    has_partition_embeddings = true;
                }
            }
        }
    }

    auto tensor_options = torch::TensorOptions().device(devices[0]).dtype(torch::kFloat32);

    if (compute_worker) {
        encoder = std::make_shared<GeneralEncoder>(model_config->encoder, devices[0], num_relations, num_partitions);

        if (model_config->learning_task == LearningTask::LINK_PREDICTION) {
            shared_ptr <EdgeDecoderOptions> decoder_options = std::dynamic_pointer_cast<EdgeDecoderOptions>(
                    model_config->decoder->options);

            int last_stage = model_config->encoder->layers.size() - 1;
            int last_layer = model_config->encoder->layers[last_stage].size() - 1;
            int64_t dim = model_config->encoder->layers[last_stage][last_layer]->output_dim;

            decoder = get_edge_decoder(model_config->decoder->type, decoder_options->edge_decoder_method, num_relations,
                                       dim, tensor_options,
                                       decoder_options->inverse_edges);
        } else {
            decoder = get_node_decoder(model_config->decoder->type);
        }

        loss = getLossFunction(model_config->loss);

        model = std::make_shared<Model>(encoder, decoder, loss);
    } else {
        model = std::make_shared<Model>(encoder, decoder, loss, nullptr, model_config->learning_task);
    }

    model->device_ = devices[0];
    model->device_models_ = std::vector<shared_ptr<Model>>(devices.size());
//    model->device_models_ = std::vector<shared_ptr<Model>>(2);

    if (train and compute_worker) {
        model->setup_optimizers(model_config);

        if (model_config->sparse_optimizer != nullptr) {
            model->sparse_lr_ = model_config->sparse_optimizer->options->learning_rate;
        } else {
            model->sparse_lr_ = model_config->dense_optimizer->options->learning_rate;
        }
    }

    model->model_config_ = model_config;

    if (devices.size() > 1 and compute_worker) {
        SPDLOG_INFO("Broadcasting model to: {} GPUs", devices.size());
        model->createDeviceModels(devices);
    } else {
        model->device_models_[0] = model;
    }

//    model->model_config_ = model_config;
//    std::vector<torch::Device> tmp = {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)};
//    model->broadcast(tmp);


    model->has_embeddings_ = has_embeddings;
    model->has_partition_embeddings_ = has_partition_embeddings;

    model->pg_gloo_ = nullptr;
    model->dist_config_ = nullptr;
    model->dist_ = false;
    model->compute_pg_ = nullptr;
    model->compute_pg_nccl_ = nullptr;
    model->total_compute_pg_ = nullptr;
    model->batch_worker_ = false;
    model->compute_worker_ = compute_worker;
    model->first_epoch_ = true;
    model->pg_lock_ = new std::mutex();
    model->update_feeders_lock_ = new std::mutex();
    model->last_compute_worker_ = -1;
    model->already_notified_ = false;

    std::cout<<"init model"<<"\n"; // init some model listening queues here if desired for distributed training

    // TODO: we need to create the model listening queues here if we are doing that based on the distributed config


    return model;
}