//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_CONFIG_H
#define MARIUS_CONFIG_H

#include "common/datatypes.h"
#include "common/pybind_headers.h"
#include "constants.h"
#include "options.h"

using pyobj = pybind11::object;
using std::shared_ptr;

struct NeighborSamplingConfig {
    NeighborSamplingLayer type;
    shared_ptr<NeighborSamplingOptions> options = nullptr;
    bool use_hashmap_sets;
};

struct OptimizerConfig {
    OptimizerType type;
    shared_ptr<OptimizerOptions> options = nullptr;
};

struct InitConfig {
    InitDistribution type;
    shared_ptr<InitOptions> options = nullptr;

    InitConfig(){};
    InitConfig(InitDistribution type, shared_ptr<InitOptions> options) : type(type), options(options){};
};

struct LossConfig {
    LossFunctionType type;
    shared_ptr<LossOptions> options = nullptr;
};

struct LayerConfig {
    LayerType type;
    shared_ptr<LayerOptions> options = nullptr;
    int input_dim;
    int output_dim;
    shared_ptr<InitConfig> init = nullptr;
    shared_ptr<OptimizerConfig> optimizer = nullptr;
    bool bias;
    shared_ptr<InitConfig> bias_init = nullptr;
    ActivationFunction activation;
};

struct EncoderConfig {
    bool use_incoming_nbrs;
    bool use_outgoing_nbrs;
    std::vector<std::vector<shared_ptr<LayerConfig>>> layers;
    std::vector<shared_ptr<NeighborSamplingConfig>> train_neighbor_sampling;
    std::vector<shared_ptr<NeighborSamplingConfig>> eval_neighbor_sampling;
};

struct DecoderConfig {
    DecoderType type;
    shared_ptr<DecoderOptions> options = nullptr;
    shared_ptr<OptimizerConfig> optimizer = nullptr;
};

struct StorageBackendConfig {
    StorageBackend type;
    shared_ptr<StorageOptions> options = nullptr;
};

struct DatasetConfig {
    string dataset_dir;
    int64_t num_edges;
    int64_t num_nodes;
    int64_t num_relations;
    int64_t num_train;
    int64_t num_valid;
    int64_t num_test;
    int node_feature_dim;
    int rel_feature_dim;
    int num_classes;
};

struct NegativeSamplingConfig {
    int num_chunks;
    int negatives_per_positive;
    float degree_fraction;
    bool filtered;
    LocalFilterMode local_filter_mode;
};

struct PipelineConfig {
    bool sync;
    int staleness_bound;
    int gpu_sync_interval;
    bool gpu_model_average;
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

struct CheckpointConfig {
    // TODO: save the checkpoint which performs best on the valid/test set.
    bool save_best;
    int interval;
    bool save_state;
};

struct ModelConfig {
    int random_seed;
    LearningTask learning_task;
    shared_ptr<EncoderConfig> encoder = nullptr;
    shared_ptr<DecoderConfig> decoder = nullptr;
    shared_ptr<LossConfig> loss = nullptr;
    shared_ptr<OptimizerConfig> dense_optimizer = nullptr;
    shared_ptr<OptimizerConfig> sparse_optimizer = nullptr;
};

struct StorageConfig {
    torch::Device device_type = torch::Device("cpu");
    std::vector<int> device_ids = {};
    shared_ptr<DatasetConfig> dataset = nullptr;
    shared_ptr<StorageBackendConfig> edges = nullptr;
    shared_ptr<StorageBackendConfig> nodes = nullptr;
    shared_ptr<StorageBackendConfig> embeddings = nullptr;
    shared_ptr<StorageBackendConfig> features = nullptr;
    bool prefetch;
    bool shuffle_input;
    bool full_graph_evaluation;
    bool export_encoded_nodes;
    std::string model_dir;
    spdlog::level::level_enum log_level;
    bool train_edges_pre_sorted;
};

struct TrainingConfig {
    int batch_size;
    shared_ptr<NegativeSamplingConfig> negative_sampling = nullptr;
    int num_epochs;
    shared_ptr<PipelineConfig> pipeline = nullptr;
    int epochs_per_shuffle;
    int logs_per_epoch;
    bool save_model;
    shared_ptr<CheckpointConfig> checkpoint = nullptr;
    bool resume_training;
    string resume_from_checkpoint;
};

struct EvaluationConfig {
    int batch_size;
    shared_ptr<NegativeSamplingConfig> negative_sampling = nullptr;
    shared_ptr<PipelineConfig> pipeline = nullptr;
    int epochs_per_eval;
    string checkpoint_dir;
    bool full_graph_evaluation;
};

struct MariusConfig {
    shared_ptr<ModelConfig> model = nullptr;
    shared_ptr<StorageConfig> storage = nullptr;
    shared_ptr<TrainingConfig> training = nullptr;
    shared_ptr<EvaluationConfig> evaluation = nullptr;
};

bool check_missing(pyobj python_object);

template <typename T>
T cast_helper(pyobj python_object);

PYBIND11_EXPORT shared_ptr<NeighborSamplingConfig> initNeighborSamplingConfig(pyobj python_object);

// Lol at this name
PYBIND11_EXPORT shared_ptr<InitConfig> initInitConfig(pyobj python_object);

PYBIND11_EXPORT shared_ptr<OptimizerConfig> initOptimizerConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<DatasetConfig> initDatasetConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<LayerConfig> initLayerConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<EncoderConfig> initEncoderConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<DecoderConfig> initDecoderConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<LossConfig> initLossConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<StorageBackendConfig> initStorageBackendConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<NegativeSamplingConfig> initNegativeSamplingConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<PipelineConfig> initPipelineConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<CheckpointConfig> initCheckpointConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<ModelConfig> initModelConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<StorageConfig> initStorageConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<TrainingConfig> initTrainingConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<EvaluationConfig> initEvaluationConfig(pyobj python_config);

PYBIND11_EXPORT shared_ptr<MariusConfig> initMariusConfig(pyobj python_config);

shared_ptr<MariusConfig> loadConfig(string config_path, bool save = false);

#endif  // MARIUS_CONFIG_H
