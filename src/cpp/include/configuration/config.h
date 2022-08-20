//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_CONFIG_H
#define MARIUS_CONFIG_H

#include <pybind11/embed.h>

#include "constants.h"
#include "datatypes.h"
#include "options.h"

using pyobj = pybind11::object;
using std::shared_ptr;

struct NeighborSamplingConfig {
    NeighborSamplingLayer type;
    shared_ptr<NeighborSamplingOptions> options;
};

struct OptimizerConfig {
    OptimizerType type;
    shared_ptr<OptimizerOptions> options;
};

struct InitConfig {
    InitDistribution type;
    shared_ptr<InitOptions> options;
};

struct LossConfig {
    LossFunctionType type;
    shared_ptr<LossOptions> options;
};

struct EmbeddingsConfig {
    int dimension;
    shared_ptr<InitConfig> init;
    shared_ptr<OptimizerConfig> optimizer;
};

struct FeaturizerConfig {
    FeaturizerType type;
    shared_ptr<FeaturizerOptions> options;
    shared_ptr<OptimizerConfig> optimizer;
};

struct GNNLayerConfig {
    shared_ptr<NeighborSamplingConfig> train_neighbor_sampling;
    shared_ptr<NeighborSamplingConfig> eval_neighbor_sampling;
    shared_ptr<InitConfig> init;
    GNNLayerType type;
    shared_ptr<GNNLayerOptions> options;
    ActivationFunction activation;
    bool bias;
    shared_ptr<InitConfig> bias_init;
};

struct EncoderConfig {
    int input_dim;
    int output_dim;
    std::vector<shared_ptr<GNNLayerConfig>> layers;
    shared_ptr<OptimizerConfig> optimizer;
    bool use_incoming_nbrs;
    bool use_outgoing_nbrs;
    bool use_hashmap_sets;
};

struct DecoderConfig {
    DecoderType type;
    shared_ptr<DecoderOptions> options;
    shared_ptr<OptimizerConfig> optimizer;
};

struct StorageBackendConfig {
    StorageBackend type;
    shared_ptr<StorageOptions> options;
};

struct DatasetConfig {
    string base_directory;
    int64_t num_edges;
    int64_t num_nodes;
    int64_t num_relations;
    int64_t num_train;
    int64_t num_valid;
    int64_t num_test;
    int feature_dim;
    int num_classes;
};

struct NegativeSamplingConfig {
    int num_chunks;
    int negatives_per_positive;
    float degree_fraction;
    bool filtered;
};

struct PipelineConfig {
    bool sync;
    int staleness_bound;
    int batch_host_queue_size;
    int batch_device_queue_size;
    int gradients_device_queue_size;
    int gradients_host_queue_size;
    int batch_loader_threads;
    int batch_transfer_threads;
    int compute_threads;
    int gradient_transfer_threads;
    int gradient_update_threads;
};

struct ModelConfig {
    int random_seed;
    LearningTask learning_task;
    shared_ptr<EmbeddingsConfig> embeddings;
    shared_ptr<FeaturizerConfig> featurizer;
    shared_ptr<EncoderConfig> encoder;
    shared_ptr<DecoderConfig> decoder;
    shared_ptr<LossConfig> loss;
};

struct StorageConfig {
    torch::DeviceType device_type;
    std::vector<int> device_ids;
    shared_ptr<DatasetConfig> dataset;
    shared_ptr<StorageBackendConfig> edges;
    shared_ptr<StorageBackendConfig> nodes;
    shared_ptr<StorageBackendConfig> embeddings;
    shared_ptr<StorageBackendConfig> features;
    bool prefetch;
    bool shuffle_input;
    bool full_graph_evaluation;
};

struct TrainingConfig {
    int batch_size;
    shared_ptr<NegativeSamplingConfig> negative_sampling;
    int num_epochs;
    shared_ptr<PipelineConfig> pipeline;
    int epochs_per_shuffle;
    int logs_per_epoch;
};

struct EvaluationConfig {
    int batch_size;
    shared_ptr<NegativeSamplingConfig> negative_sampling;
    shared_ptr<PipelineConfig> pipeline;
    int epochs_per_eval;
    int eval_checkpoint;
    bool full_graph_evaluation;
};

struct MariusConfig {
    shared_ptr<ModelConfig> model;
    shared_ptr<StorageConfig> storage;
    shared_ptr<TrainingConfig> training;
    shared_ptr<EvaluationConfig> evaluation;
};

bool check_missing(pyobj python_object);

template <typename T>
T cast_helper(pyobj python_object);

shared_ptr<NeighborSamplingConfig> initNeighborSamplingConfig(pyobj python_object);

// Lol at this name
shared_ptr<InitConfig> initInitConfig(pyobj python_object);

shared_ptr<OptimizerConfig> initOptimizerConfig(pyobj python_config);

shared_ptr<DatasetConfig> initDatasetConfig(pyobj python_config);

shared_ptr<EmbeddingsConfig> initEmbeddingsConfig(pyobj python_config);

shared_ptr<FeaturizerConfig> initFeaturizerConfig(pyobj python_config);

shared_ptr<GNNLayerConfig> initGNNLayerConfig(pyobj python_config);

shared_ptr<EncoderConfig> initEncoderConfig(pyobj python_config);

shared_ptr<DecoderConfig> initDecoderConfig(pyobj python_config);

shared_ptr<LossConfig> initLossConfig(pyobj python_config);

shared_ptr<StorageBackendConfig> initStorageBackendConfig(pyobj python_config);

shared_ptr<NegativeSamplingConfig> initNegativeSamplingConfig(pyobj python_config);

shared_ptr<PipelineConfig> initPipelineConfig(pyobj python_config);

shared_ptr<ModelConfig> initModelConfig(pyobj python_config);

shared_ptr<StorageConfig> initStorageConfig(pyobj python_config);

shared_ptr<TrainingConfig> initTrainingConfig(pyobj python_config);

shared_ptr<EvaluationConfig> initEvaluationConfig(pyobj python_config);

shared_ptr<MariusConfig> initMariusConfig(pyobj python_config);

shared_ptr<MariusConfig> initConfig(string config_path = "");

#endif //MARIUS_CONFIG_H
