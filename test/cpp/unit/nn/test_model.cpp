//
// Created by Jason Mohoney on 2/5/22.
//

#include <gtest/gtest.h>
#include <nn/decoders/edge/corrupt_node_decoder.h>
#include <nn/decoders/edge/corrupt_rel_decoder.h>
#include <nn/decoders/edge/distmult.h>
#include <nn/decoders/edge/transe.h>
#include <nn/decoders/node/node_decoder.h>
#include <nn/decoders/node/node_decoder_model.h>
#include <nn/decoders/node/noop_node_decoder.h>
#include <nn/layers/embedding/embedding.h>
#include <nn/layers/feature/feature.h>
#include <nn/model.h>

TEST(TestModel, TestInitModelFromConfigLP) {
    int embedding_dim = 50;
    int random_seed = 100;
    int num_relations = 10;

    auto model_config = std::make_shared<ModelConfig>();
    auto encoder_config = std::make_shared<EncoderConfig>();
    auto decoder_config = std::make_shared<DecoderConfig>();
    auto loss_config = std::make_shared<LossConfig>();
    auto dense_optimizer = std::make_shared<OptimizerConfig>();
    auto sparse_optimizer = std::make_shared<OptimizerConfig>();

    dense_optimizer->type = OptimizerType::SGD;
    sparse_optimizer->type = OptimizerType::SGD;

    auto optimizer_options = std::make_shared<OptimizerOptions>();
    optimizer_options->learning_rate = .1;

    dense_optimizer->options = optimizer_options;
    sparse_optimizer->options = optimizer_options;

    auto layer_config = std::make_shared<LayerConfig>();
    layer_config->type = LayerType::EMBEDDING;
    layer_config->output_dim = embedding_dim;

    std::vector<shared_ptr<LayerConfig>> stage;
    stage.emplace_back(layer_config);
    encoder_config->layers.emplace_back(stage);

    decoder_config->type = DecoderType::DISTMULT;
    auto decoder_options = std::make_shared<EdgeDecoderOptions>();
    decoder_options->inverse_edges = true;
    decoder_options->edge_decoder_method = EdgeDecoderMethod::CORRUPT_NODE;
    decoder_config->options = decoder_options;

    loss_config->type = LossFunctionType::SOFTMAX_CE;
    auto loss_options = std::make_shared<LossOptions>();
    loss_options->loss_reduction = LossReduction::SUM;
    loss_config->options = loss_options;

    model_config->random_seed = random_seed;
    model_config->learning_task = LearningTask::LINK_PREDICTION;

    // check missing encoder config
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, num_relations, true), UnexpectedNullPtrException);
    model_config->encoder = encoder_config;

    // check missing decoder config
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, num_relations, true), UnexpectedNullPtrException);
    model_config->decoder = decoder_config;

    // check missing loss
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, num_relations, true), UnexpectedNullPtrException);
    model_config->loss = loss_config;

    // check missing dense optimizer
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, num_relations, true), UnexpectedNullPtrException);
    model_config->dense_optimizer = dense_optimizer;
    model_config->sparse_optimizer = sparse_optimizer;

    ASSERT_NO_THROW(initModelFromConfig(model_config, {torch::kCPU}, num_relations, true));
    shared_ptr<Model> model = initModelFromConfig(model_config, {torch::kCPU}, num_relations, true);

    // check learning task
    ASSERT_EQ(model->learning_task_, LearningTask::LINK_PREDICTION);

    // test encoder
    ASSERT_EQ(model->encoder_->encoder_config_->train_neighbor_sampling.size(), 0);
    ASSERT_EQ(model->encoder_->encoder_config_->eval_neighbor_sampling.size(), 0);
    ASSERT_EQ(model->encoder_->layers_.size(), 1);
    ASSERT_EQ(model->encoder_->layers_[0].size(), 1);

    bool is_instance = instance_of<Layer, EmbeddingLayer>(model->encoder_->layers_[0][0]);
    bool is_not_instance = !instance_of<Layer, FeatureLayer>(model->encoder_->layers_[0][0]);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // test link prediction model decoder
    is_instance = instance_of<DecoderModel, DistMult>(model->decoder_->model_);
    is_not_instance = !instance_of<DecoderModel, TransE>(model->decoder_->model_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    is_instance = instance_of<Decoder, CorruptNodeDecoder>(model->decoder_);
    is_not_instance = !instance_of<Decoder, CorruptRelDecoder>(model->decoder_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // test loss
    is_instance = instance_of<LossFunction, SoftmaxCrossEntropy>(model->loss_function_);
    is_not_instance = !instance_of<LossFunction, SoftPlusLoss>(model->loss_function_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // check optimizers set properly
    ASSERT_EQ(model->optimizers_.size(), 1);

    is_instance = instance_of<Optimizer, SGDOptimizer>(model->optimizers_[0]);
    is_not_instance = !instance_of<Optimizer, AdagradOptimizer>(model->optimizers_[0]);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);
}

TEST(TestModel, TestInitModelFromConfigNC) {
    int feature_dim = 50;
    int random_seed = 100;

    auto model_config = std::make_shared<ModelConfig>();
    auto encoder_config = std::make_shared<EncoderConfig>();
    auto decoder_config = std::make_shared<DecoderConfig>();
    auto loss_config = std::make_shared<LossConfig>();
    auto dense_optimizer = std::make_shared<OptimizerConfig>();
    auto sparse_optimizer = std::make_shared<OptimizerConfig>();

    dense_optimizer->type = OptimizerType::SGD;
    sparse_optimizer->type = OptimizerType::SGD;

    auto optimizer_options = std::make_shared<OptimizerOptions>();
    optimizer_options->learning_rate = .1;

    dense_optimizer->options = optimizer_options;
    sparse_optimizer->options = optimizer_options;

    auto layer_config = std::make_shared<LayerConfig>();
    layer_config->type = LayerType::FEATURE;
    layer_config->output_dim = feature_dim;

    std::vector<shared_ptr<LayerConfig>> stage;
    stage.emplace_back(layer_config);
    encoder_config->layers.emplace_back(stage);

    decoder_config->type = DecoderType::NODE;

    loss_config->type = LossFunctionType::SOFTMAX_CE;
    auto loss_options = std::make_shared<LossOptions>();
    loss_options->loss_reduction = LossReduction::SUM;
    loss_config->options = loss_options;

    model_config->random_seed = random_seed;
    model_config->learning_task = LearningTask::NODE_CLASSIFICATION;

    // check missing encoder config
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, -1, true), UnexpectedNullPtrException);
    model_config->encoder = encoder_config;

    // check missing decoder config
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, -1, true), UnexpectedNullPtrException);
    model_config->decoder = decoder_config;

    // check missing loss
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, -1, true), UnexpectedNullPtrException);
    model_config->loss = loss_config;

    // check missing dense optimizer
    ASSERT_THROW(initModelFromConfig(model_config, {torch::kCPU}, -1, true), UnexpectedNullPtrException);
    model_config->dense_optimizer = dense_optimizer;
    model_config->sparse_optimizer = sparse_optimizer;

    ASSERT_NO_THROW(initModelFromConfig(model_config, {torch::kCPU}, -1, true));
    shared_ptr<Model> model = initModelFromConfig(model_config, {torch::kCPU}, -1, true);

    // check learning task
    ASSERT_EQ(model->learning_task_, LearningTask::NODE_CLASSIFICATION);

    bool is_instance = instance_of<Layer, FeatureLayer>(model->encoder_->layers_[0][0]);
    bool is_not_instance = !instance_of<Layer, EmbeddingLayer>(model->encoder_->layers_[0][0]);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // test link prediction model decoder
    is_instance = instance_of<DecoderModel, NoOpNodeDecoder>(model->decoder_->model_);
    is_not_instance = !instance_of<DecoderModel, TransE>(model->decoder_->model_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    is_instance = instance_of<Decoder, NodeDecoder>(model->decoder_);
    is_not_instance = !instance_of<Decoder, CorruptNodeDecoder>(model->decoder_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // test loss
    is_instance = instance_of<LossFunction, SoftmaxCrossEntropy>(model->loss_function_);
    is_not_instance = !instance_of<LossFunction, SoftPlusLoss>(model->loss_function_);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);

    // check optimizers set properly
    ASSERT_EQ(model->optimizers_.size(), 1);

    is_instance = instance_of<Optimizer, SGDOptimizer>(model->optimizers_[0]);
    is_not_instance = !instance_of<Optimizer, AdagradOptimizer>(model->optimizers_[0]);

    ASSERT_TRUE(is_instance);
    ASSERT_TRUE(is_not_instance);
}