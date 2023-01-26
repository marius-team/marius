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
             std::vector<shared_ptr<Optimizer>> optimizers)
    : device_(torch::Device(torch::kCPU)) {
    encoder_ = encoder;
    decoder_ = decoder;
    loss_function_ = loss;
    reporter_ = reporter;
    optimizers_ = optimizers;
    learning_task_ = decoder_->learning_task_;

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

    for (int i = 0; i < named_parameters().keys().size(); i++) {
        string key = named_parameters().keys()[i];

        std::vector<torch::Tensor> input_gradients(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            if (!device_models_[j]->named_parameters()[key].mutable_grad().defined()) {
                device_models_[j]->named_parameters()[key].mutable_grad() = torch::zeros_like(device_models_[j]->named_parameters()[key]);
            }

            input_gradients[j] = (device_models_[j]->named_parameters()[key].mutable_grad());
        }

#ifdef MARIUS_CUDA
        torch::cuda::nccl::all_reduce(input_gradients, input_gradients);
#endif
    }

    step_all();
    clear_grad_all();
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
    int max_offset = 0;
    int size = 0;

    for (auto stage : encoder_->layers_) {
        for (auto layer : stage) {
            if (layer->config_->type == LayerType::EMBEDDING) {
                int offset = std::dynamic_pointer_cast<EmbeddingLayer>(layer)->offset_;

                if (size == 0) {
                    size = layer->config_->output_dim;
                }

                if (offset > max_offset) {
                    max_offset = offset;
                    size = layer->config_->output_dim;
                }
            }
        }
    }

    return max_offset + size;
}

bool Model::has_embeddings() { return encoder_->has_embeddings_; }

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

    if (call_step) {
        step();
    }

    if (batch->node_embeddings_.defined()) {
        batch->accumulateGradients(sparse_lr_);
    }
}

void Model::evaluate_batch(shared_ptr<Batch> batch) {
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto all_scores = forward_lp(batch, true);
        torch::Tensor pos_scores = std::get<0>(all_scores);
        torch::Tensor neg_scores = std::get<1>(all_scores);
        torch::Tensor inv_pos_scores = std::get<2>(all_scores);
        torch::Tensor inv_neg_scores = std::get<3>(all_scores);

        if (neg_scores.defined()) {
            std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(pos_scores, neg_scores);
        }

        if (inv_neg_scores.defined()) {
            std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(inv_pos_scores, inv_neg_scores);
        }
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        torch::Tensor y_pred = forward_nc(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, true);
        torch::Tensor labels = batch->node_labels_;

        std::dynamic_pointer_cast<NodeClassificationReporter>(reporter_)->addResult(labels, y_pred);

    } else {
        throw MariusRuntimeException("Unsupported learning task for evaluation");
    }
}

void Model::broadcast(std::vector<torch::Device> devices) {
    int i = 0;
    for (auto device : devices) {
        SPDLOG_INFO("Broadcast to GPU {}", device.index());
        if (device != device_) {
            shared_ptr<GeneralEncoder> encoder = encoder_clone_helper(encoder_, device);
            shared_ptr<Decoder> decoder = decoder_clone_helper(decoder_, device);
            device_models_[i] = std::make_shared<Model>(encoder, decoder, loss_function_, reporter_);

            for (auto optim : optimizers_) {
                device_models_[i]->optimizers_.emplace_back(optim->clone());
                device_models_[i]->sparse_lr_ = sparse_lr_;
            }
        } else {
            device_models_[i] = std::dynamic_pointer_cast<Model>(shared_from_this());
        }
        i++;
    }
}

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, bool train) {
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

    auto tensor_options = torch::TensorOptions().device(devices[0]).dtype(torch::kFloat32);

    encoder = std::make_shared<GeneralEncoder>(model_config->encoder, devices[0], num_relations);

    if (model_config->learning_task == LearningTask::LINK_PREDICTION) {
        shared_ptr<EdgeDecoderOptions> decoder_options = std::dynamic_pointer_cast<EdgeDecoderOptions>(model_config->decoder->options);

        int last_stage = model_config->encoder->layers.size() - 1;
        int last_layer = model_config->encoder->layers[last_stage].size() - 1;
        int64_t dim = model_config->encoder->layers[last_stage][last_layer]->output_dim;

        decoder = get_edge_decoder(model_config->decoder->type, decoder_options->edge_decoder_method, num_relations, dim, tensor_options,
                                   decoder_options->inverse_edges);
    } else {
        decoder = get_node_decoder(model_config->decoder->type);
    }

    loss = getLossFunction(model_config->loss);

    model = std::make_shared<Model>(encoder, decoder, loss);
    model->device_ = devices[0];
    model->device_models_ = std::vector<shared_ptr<Model>>(devices.size());

    if (train) {
        model->setup_optimizers(model_config);

        if (model_config->sparse_optimizer != nullptr) {
            model->sparse_lr_ = model_config->sparse_optimizer->options->learning_rate;
        } else {
            model->sparse_lr_ = model_config->dense_optimizer->options->learning_rate;
        }
    }

    if (devices.size() > 1) {
        SPDLOG_INFO("Broadcasting model to: {} GPUs", devices.size());
        model->broadcast(devices);
    } else {
        model->device_models_[0] = model;
    }

    return model;
}