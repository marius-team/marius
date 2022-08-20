//
// Created by Jason Mohoney on 2/12/21.
//

#include "model.h"

#ifdef MARIUS_CUDA
    #include <torch/csrc/cuda/nccl.h>
#endif

#include "configuration/constants.h"
#include "logger.h"
#include "model_helpers.h"

Model::Model() : current_device_(torch::Device(torch::kCPU)) {
    current_device_ = torch::Device(torch::kCPU);
}

Model::Model(shared_ptr<ModelConfig> model_config, torch::Device device) : current_device_(torch::Device(torch::kCPU)) {
    current_device_ = device;
    model_config_ = model_config;
}

void Model::zero_grad() {

    if (featurizer_optimizer_ != nullptr) {
        featurizer_optimizer_->zero_grad();
    }

    if (encoder_optimizer_ != nullptr) {
        encoder_optimizer_->zero_grad();
    }

    if (decoder_optimizer_ != nullptr) {
        decoder_optimizer_->zero_grad();
    }
}

void Model::step() {

    if (featurizer_optimizer_ != nullptr) {
        featurizer_optimizer_->step();
    }

    if (encoder_optimizer_ != nullptr) {
        encoder_optimizer_->step();
    }

    if (decoder_optimizer_ != nullptr) {
        decoder_optimizer_->step();
    }
}


void Model::save(std::string directory) {

    string model_filename = directory + PathConstants::model_file;

    // TODO saving optimizer state seems to be bugged in the c++ api of Pytorch. optimizer.step() fails
    string model_state_filename = directory + PathConstants::model_state_file;

    torch::serialize::OutputArchive model_archive;

    bool will_save = false;

    if (featurizer_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(featurizer_)->save(model_archive);
        will_save = true;
    }

    if (encoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->save(model_archive);
        will_save = true;
    }

    if (decoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->save(model_archive);
        will_save = true;
    }

    if (will_save) {
        model_archive.save_to(model_filename);
    }
}

void Model::load(std::string directory) {

    string model_filename = directory + PathConstants::model_file;

    // TODO saving optimizer state seems to be bugged in the c++ api of Pytorch. optimizer.step() fails
    string model_state_filename = directory + PathConstants::model_state_file;

    torch::serialize::InputArchive model_archive;
    model_archive.load_from(model_filename);

    if (featurizer_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(featurizer_)->load(model_archive);
    }

    if (encoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->load(model_archive);
    }

    if (decoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->load(model_archive);
    }
}

std::shared_ptr<Model> Model::clone_to_device(torch::Device device) {
    std::shared_ptr<Model> primary_model = std::dynamic_pointer_cast<Model>(shared_from_this());

    if (device != current_device_) {
        std::shared_ptr<Model> cloned_model = std::make_shared<Model>(model_config_, device);

        if (featurizer_ != nullptr) {
            cloned_model->featurizer_ = std::dynamic_pointer_cast<Featurizer>(featurizer_->clone(device));
            cloned_model->register_module<Featurizer>("featurizer", cloned_model->featurizer_);
//            cloned_model->featurizer_optimizer_ = getOptimizerForModule(cloned_model->featurizer_, model_config_.featurizer.);
        }

        if (encoder_ != nullptr) {
            if (!model_config_->encoder->layers.empty()) {
                encoder_clone_helper<GeneralGNN>(primary_model, cloned_model, device);
            }
        }

        if (decoder_ != nullptr) {
            if (model_config_->decoder->type != DecoderType::NONE) {
                if (model_config_->decoder->type == DecoderType::DISTMULT) {
                    decoder_clone_helper<DistMult>(primary_model, cloned_model, device);
                } else if (model_config_->decoder->type == DecoderType::TRANSE) {
                    decoder_clone_helper<TransE>(primary_model, cloned_model, device);
                } else if (model_config_->decoder->type == DecoderType::COMPLEX) {
                    decoder_clone_helper<ComplEx>(primary_model, cloned_model, device);
                } else {
                    throw std::runtime_error("Decoder currently not supported.");
                }
            }
        }

        cloned_model->loss_function_ = getLossFunction(model_config_->loss);
        cloned_model->reporter_ = reporter_;
        cloned_model->devices_ = devices_;

        return cloned_model;
    } else {
        return primary_model;
    }
}

void Model::broadcast(std::vector<torch::Device> devices) {
    device_models_ = std::vector<std::shared_ptr<Model>>(devices.size());

    int i = 0;
    for (auto device : devices) {
        SPDLOG_INFO("Broadcast to GPU {}", device.index());
        device_models_[i++] = std::dynamic_pointer_cast<Model>(clone_to_device(device));
    }
}

void Model::allReduce() {

    int num_gpus = device_models_.size();

    #pragma omp parallel for
    for (int i = 0; i < named_parameters().keys().size(); i++) {
        string key = named_parameters().keys()[i];
        std::vector<torch::Tensor> input_gradients(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            input_gradients[j] = (device_models_[j]->named_parameters()[key].mutable_grad());
        }

        #ifdef MARIUS_CUDA
            torch::cuda::nccl::all_reduce(input_gradients, input_gradients);
        #endif
    }
}

NodeClassificationModel::NodeClassificationModel(shared_ptr<ModelConfig> model_config, shared_ptr<GeneralGNN> encoder, shared_ptr<LossFunction> loss, shared_ptr<Regularizer> regularizer, shared_ptr<Featurizer> featurizer, shared_ptr<Reporter> reporter) {
    featurizer_ = featurizer;
    encoder_ = encoder;
    decoder_ = nullptr;
    loss_function_ = loss;
    regularizer_ = regularizer;
    reporter_ = reporter;
    model_config_ = model_config;
    learning_task_ = model_config_->learning_task;

    featurizer_optimizer_ = nullptr;
    encoder_optimizer_ = nullptr;
    decoder_optimizer_ = nullptr;

    if (reporter_ == nullptr) {
        reporter_ = std::make_shared<NodeClassificationReporter>();
        reporter_->addMetric(new CategoricalAccuracyMetric);
    }

    if (featurizer_ != nullptr) {
        register_module<Featurizer>("featurizer", featurizer_);
        featurizer_optimizer_ = getOptimizerForModule(featurizer_, model_config_->featurizer->optimizer);
    }

    if (encoder_ != nullptr) {
        register_module("encoder", std::dynamic_pointer_cast<torch::nn::Module>(encoder_));
        encoder_optimizer_ = getOptimizerForModule(std::dynamic_pointer_cast<torch::nn::Module>(encoder_), model_config_->encoder->optimizer);
    }
}

Labels NodeClassificationModel::forward(Batch *batch, bool train) {

    if (train && featurizer_ != nullptr) {
        batch->unique_node_embeddings_.requires_grad_();
    }

    Embeddings inputs;
    if (featurizer_ != nullptr) {
        inputs = featurizer_->operator()(batch->unique_node_features_, batch->unique_node_embeddings_);
    } else {
        inputs = batch->unique_node_features_;
    }

    batch->gnn_graph_.performMap();
    return encoder_->forward(inputs, batch->gnn_graph_, train);
}

void NodeClassificationModel::train_batch(Batch *batch) {

    zero_grad();

    Labels y_predicted = forward(batch, true);
    torch::Tensor targets = batch->unique_node_labels_.to(torch::kInt64).flatten(0, 1);
    torch::Tensor loss = torch::nn::functional::cross_entropy(y_predicted, targets);

    loss.backward();

    step();
}

void NodeClassificationModel::train_batch(std::vector<Batch *> sub_batches) {
    zero_grad();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < devices_.size(); i++) {
            Labels y_predicted = ((NodeClassificationModel *) device_models_[i].get())->forward(sub_batches[i], true);
            Labels y_true = sub_batches[i]->unique_node_labels_;
            torch::Tensor targets = torch::argmax(y_true, 1);

            torch::Tensor loss = torch::nn::functional::cross_entropy(y_predicted, targets);

            loss.backward();
        }
    }
    allReduce();
    step();
}

void NodeClassificationModel::evaluate(Batch *batch, bool filtered_eval) {
    Labels y_predicted = forward(batch, false);
    Labels y_true = batch->unique_node_labels_.to(torch::kInt64).flatten(0, 1);
    std::dynamic_pointer_cast<NodeClassificationReporter>(reporter_)->addResult(y_true, y_predicted);
}

void NodeClassificationModel::evaluate(std::vector<Batch *> sub_batches, bool filtered_eval) {
    #pragma omp parallel for
    for (int i = 0; i < devices_.size(); i++) {
        Labels y_predicted = ((NodeClassificationModel *) device_models_[i].get())->forward(sub_batches[i], false);
        Labels y_true = sub_batches[i]->unique_node_labels_;
        std::dynamic_pointer_cast<NodeClassificationReporter>(reporter_)->addResult(y_true, y_predicted);
    }
}

LinkPredictionModel::LinkPredictionModel(shared_ptr<ModelConfig> model_config, shared_ptr<GeneralGNN> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Regularizer> regularizer, shared_ptr<Featurizer> featurizer, shared_ptr<Reporter> reporter) {
    featurizer_ = featurizer;
    encoder_ = encoder;
    decoder_ = decoder;
    loss_function_ = loss;
    regularizer_ = regularizer;
    reporter_ = reporter;

    model_config_ = model_config;
    learning_task_ = model_config_->learning_task;

    featurizer_optimizer_ = nullptr;
    encoder_optimizer_ = nullptr;
    decoder_optimizer_ = nullptr;

    if (reporter_ == nullptr) {
        reporter_ = std::make_shared<LinkPredictionReporter>();
        reporter_->addMetric(new MeanRankMetric);
        reporter_->addMetric(new MeanReciprocalRankMetric);
        reporter_->addMetric(new HitskMetric(1));
        reporter_->addMetric(new HitskMetric(3));
        reporter_->addMetric(new HitskMetric(5));
        reporter_->addMetric(new HitskMetric(10));
        reporter_->addMetric(new HitskMetric(50));
        reporter_->addMetric(new HitskMetric(100));
    }

    if (featurizer_ != nullptr) {
        register_module<Featurizer>("featurizer", featurizer_);
        featurizer_optimizer_ = getOptimizerForModule(featurizer_, model_config_->featurizer->optimizer);
    }

    if (encoder_ != nullptr) {
        register_module("encoder", std::dynamic_pointer_cast<torch::nn::Module>(encoder_));
        encoder_optimizer_ = getOptimizerForModule(std::dynamic_pointer_cast<torch::nn::Module>(encoder_), model_config_->encoder->optimizer);
    }

    if (decoder_ != nullptr) {
        register_module("decoder", std::dynamic_pointer_cast<torch::nn::Module>(decoder_));
        decoder_optimizer_ = getOptimizerForModule(std::dynamic_pointer_cast<torch::nn::Module>(decoder_), model_config_->decoder->optimizer);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LinkPredictionModel::forward(Batch *batch, bool train) {

    if (train) {
        batch->unique_node_embeddings_.requires_grad_();
    }

    Embeddings inputs;
    if (featurizer_ != nullptr) {
        inputs = (*featurizer_)(batch->unique_node_features_, batch->unique_node_embeddings_);
    } else {
        inputs = batch->unique_node_embeddings_;
    }

    if (model_config_->encoder != nullptr) {
        batch->gnn_graph_.performMap();
        batch->encoded_uniques_ = encoder_->forward(inputs, batch->gnn_graph_, train);
    }

    batch->prepareBatch();

    return decoder_->forward(batch, train);
}

void LinkPredictionModel::train_batch(Batch *batch) {

    zero_grad();

    auto all_scores = forward(batch, true);
    torch::Tensor rhs_pos_scores = std::get<0>(all_scores);
    torch::Tensor rhs_neg_scores = std::get<1>(all_scores);
    torch::Tensor lhs_pos_scores = std::get<2>(all_scores);
    torch::Tensor lhs_neg_scores = std::get<3>(all_scores);

    torch::Tensor loss;
    if (lhs_neg_scores.defined()) {
        torch::Tensor rhs_loss = (*loss_function_)(rhs_pos_scores, rhs_neg_scores);
        torch::Tensor lhs_loss = (*loss_function_)(lhs_pos_scores, lhs_neg_scores);
        loss = lhs_loss + rhs_loss;
    } else {
        loss = (*loss_function_)(rhs_pos_scores, rhs_neg_scores);
    }

    loss.backward();

    step();
}

void LinkPredictionModel::train_batch(std::vector<Batch *> sub_batches) {

    #pragma omp parallel for
    for (int i = 0; i < devices_.size(); i++) {
        device_models_[i]->zero_grad();

        auto all_scores = ((LinkPredictionModel *) device_models_[i].get())->forward(sub_batches[i], true);
        torch::Tensor rhs_pos_scores = std::get<0>(all_scores);
        torch::Tensor rhs_neg_scores = std::get<1>(all_scores);
        torch::Tensor lhs_pos_scores = std::get<2>(all_scores);
        torch::Tensor lhs_neg_scores = std::get<3>(all_scores);

        torch::Tensor loss;
        if (lhs_neg_scores.defined()) {
            torch::Tensor rhs_loss = (*loss_function_)(rhs_pos_scores, rhs_neg_scores);
            torch::Tensor lhs_loss = (*loss_function_)(lhs_pos_scores, lhs_neg_scores);
            loss = lhs_loss + rhs_loss;
        } else {
            loss = (*loss_function_)(rhs_pos_scores, rhs_neg_scores);
        }

        loss.backward();
    }

    allReduce();

    #pragma omp parallel for
    for (int i = 0; i < devices_.size(); i++) {
        device_models_[i]->step();
    }
}


void LinkPredictionModel::evaluate(Batch *batch, bool filtered_eval) {
    auto all_scores = forward(batch, false);
    torch::Tensor rhs_pos_scores = std::get<0>(all_scores);
    torch::Tensor rhs_neg_scores = std::get<1>(all_scores);

    if (filtered_eval) {
        for (int64_t i = 0; i < batch->batch_size_; i++) {
            rhs_neg_scores[i].index_fill_(0, batch->dst_neg_filter_eval_[i].to(batch->src_pos_embeddings_.device()), -1e9);
        }
    }

    std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(rhs_pos_scores, rhs_neg_scores);

    if (std::dynamic_pointer_cast<LinkPredictionDecoder>(decoder_)->use_inverse_relations_) {
        torch::Tensor lhs_pos_scores = std::get<2>(all_scores);
        torch::Tensor lhs_neg_scores = std::get<3>(all_scores);

        if (filtered_eval) {
            for (int64_t i = 0; i < batch->batch_size_; i++) {
                lhs_neg_scores[i].index_fill_(0, batch->src_neg_filter_eval_[i].to(batch->src_pos_embeddings_.device()), -1e9);
            }
        }

        std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(lhs_pos_scores, lhs_neg_scores);
    }
}

void LinkPredictionModel::evaluate(std::vector<Batch *> sub_batches, bool filtered_eval) {
    #pragma omp parallel for
    for (int i = 0; i < devices_.size(); i++) {
        auto all_scores = ((LinkPredictionModel *) device_models_[i].get())->forward(sub_batches[i], true);
        torch::Tensor rhs_pos_scores = std::get<0>(all_scores);
        torch::Tensor rhs_neg_scores = std::get<1>(all_scores);
        torch::Tensor lhs_pos_scores = std::get<2>(all_scores);
        torch::Tensor lhs_neg_scores = std::get<3>(all_scores);

        if (filtered_eval) {
            for (int64_t i = 0; i < sub_batches[i]->batch_size_; i++) {
                lhs_neg_scores[i].index_fill_(0, sub_batches[i]->src_neg_filter_eval_[i].to(sub_batches[i]->src_pos_embeddings_.device()), -1e9);
                rhs_neg_scores[i].index_fill_(0, sub_batches[i]->dst_neg_filter_eval_[i].to(sub_batches[i]->src_pos_embeddings_.device()), -1e9);
            }
        }
        std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(rhs_pos_scores, rhs_neg_scores);
        std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(lhs_pos_scores, lhs_neg_scores);
    }
}

shared_ptr<Model> initializeModel(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations) {
    shared_ptr<Featurizer> featurizer = nullptr;
    shared_ptr<GeneralGNN> encoder = nullptr;
    shared_ptr<Decoder> decoder = nullptr;
    shared_ptr<LossFunction> loss = nullptr;
    shared_ptr<Regularizer> regularizer = nullptr;
    shared_ptr<Model> model;

    if (model_config->encoder != nullptr) {
        encoder = std::make_shared<GeneralGNN>(model_config->encoder, devices[0], num_relations);
    }

    if (model_config->learning_task == LearningTask::LINK_PREDICTION) {
        // TODO support other datatypes
        auto tensor_options = torch::TensorOptions().device(devices[0]).dtype(torch::kFloat32);

        if (model_config->decoder->type == DecoderType::NONE) {
            decoder = std::make_shared<EmptyDecoder>();
        } else if (model_config->decoder->type == DecoderType::DISTMULT) {
            decoder = std::make_shared<DistMult>(num_relations, model_config->decoder->options->input_dim, tensor_options, model_config->decoder->options->inverse_edges);
        } else if (model_config->decoder->type == DecoderType::TRANSE) {
            decoder = std::make_shared<TransE>(num_relations, model_config->decoder->options->input_dim, tensor_options, model_config->decoder->options->inverse_edges);
        } else if (model_config->decoder->type == DecoderType::COMPLEX) {
            decoder = std::make_shared<ComplEx>(num_relations, model_config->decoder->options->input_dim, tensor_options, model_config->decoder->options->inverse_edges);
        } else {
            SPDLOG_ERROR("Decoder currently not supported.");
            throw std::runtime_error("");
        }

        loss = getLossFunction(model_config->loss);

        model = std::make_shared<LinkPredictionModel>(model_config, encoder, decoder, loss, regularizer, featurizer);
        model->has_embeddings_ = true;
    } else {
        model = std::make_shared<NodeClassificationModel>(model_config, encoder, loss, regularizer, featurizer);
        model->has_embeddings_ = false;
    }

    model->reinitialize_ = true;
    model->current_device_ = devices[0];

    if (devices.size() > 1) {
        SPDLOG_INFO("Broadcasting model to: {} GPUs", devices.size() );
        model->broadcast(devices);
    } else {
        model->devices_ = {model->current_device_};
    }

    return model;
}

shared_ptr<torch::optim::Optimizer> getOptimizerForModule(shared_ptr<torch::nn::Module> module, shared_ptr<OptimizerConfig> optimizer_config) {

    OptimizerType optimizer_type = optimizer_config->type;
    switch (optimizer_type) {
        case OptimizerType::SGD: {
            return std::make_shared<torch::optim::SGD>(module->parameters(), optimizer_config->options->learning_rate);
        }
        case OptimizerType::ADAGRAD: {
            auto marius_options = std::dynamic_pointer_cast<AdagradOptions>(optimizer_config->options);
            torch::optim::AdagradOptions torch_options;
            torch_options.eps(marius_options->eps);
            torch_options.initial_accumulator_value(marius_options->init_value);
            torch_options.lr(marius_options->learning_rate);
            torch_options.lr_decay(marius_options->lr_decay);
            torch_options.weight_decay(marius_options->weight_decay);
            return std::make_shared<torch::optim::Adagrad>(module->parameters(), torch_options);
        }
        case OptimizerType::ADAM: {
            auto marius_options = std::dynamic_pointer_cast<AdamOptions>(optimizer_config->options);
            torch::optim::AdamOptions torch_options;
            torch_options.amsgrad(marius_options->amsgrad);
            torch_options.eps(marius_options->eps);
            torch_options.betas({marius_options->beta_1, marius_options->beta_2});
            torch_options.lr(marius_options->learning_rate);
            torch_options.weight_decay(marius_options->weight_decay);
            return std::make_shared<torch::optim::Adam>(module->parameters(), torch_options);
        }
        default: throw std::invalid_argument("Unrecognized optimizer type");
    }
}
