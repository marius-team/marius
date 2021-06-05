//
// Created by Jason Mohoney on 2/18/20.
//

#ifndef MARIUS_CONFIG_H
#define MARIUS_CONFIG_H

#include <string>

#include "datatypes.h"
#include "logger.h"

using std::string;
using std::mutex;

template <typename T>
struct OptInfo {
    T *var_ptr;
    T default_val;
    std::string s_section;
    std::string s_option;
    T range[2];
};

struct AllOptInfo {
    std::vector<OptInfo<std::string>> s_var_map;
    std::vector<OptInfo<int64_t>> i64_var_map;
    std::vector<OptInfo<int>> i_var_map;
    std::vector<OptInfo<float>> f_var_map;
    std::vector<OptInfo<bool>> b_var_map;
};

class TimestampAllocator {
  private:
    mutex time_lock_;

  public:
    Timestamp getTimestamp() {
        time_lock_.lock();
        auto ts = std::chrono::steady_clock::now();
        time_lock_.unlock();
        return ts;
    }
};

extern TimestampAllocator global_timestamp_allocator;

namespace PathConstants {
    const string edges_directory = "edges/";
    const string edges_train_directory = "train/";
    const string edges_file = "edges";
    const string edge_partition_offsets_file = "partition_offsets.txt";
    const string edges_validation_directory = "evaluation/";
    const string edges_test_directory = "test/";
    const string embeddings_directory = "embeddings/";
    const string relations_directory = "relations/";
    const string embeddings_file = "embeddings";
    const string src_relations_file = "src_relations";
    const string dst_relations_file = "dst_relations";
    const string src_state_file = "src_state";
    const string dst_state_file = "dst_state";
    const string state_file = "state";
    const string file_ext = ".bin";
    const string metadata_file = "metadata.json";
    const string training_stats_file = "training.json";
    const string evaluation_stats_file = "evaluation.json";
    const string custom_ordering_file = "ordering.in";
};

struct GeneralOptions {
    torch::DeviceType device;
    std::vector<int> gpu_ids;
    int64_t random_seed;
    int64_t num_train;
    int64_t num_valid;
    int64_t num_test;
    int64_t num_nodes;
    int64_t num_relations;
    string experiment_name;
};

struct ModelOptions {
    float scale_factor;
    InitializationDistribution initialization_distribution;
    int embedding_size;
    // Encoder Options
    EncoderModelType encoder_model;
    // Decoder Options
    DecoderModelType decoder_model;
    ComparatorType comparator;
    RelationOperatorType relation_operator;
};

struct StorageOptions {
    BackendType edges;
    bool reinitialize_edges;
    bool remove_preprocessed;
    bool shuffle_input_edges;
    torch::Dtype edges_dtype;
    BackendType embeddings;
    bool reinitialize_embeddings;
    BackendType relations;
    torch::Dtype embeddings_dtype;
    EdgeBucketOrdering edge_bucket_ordering;
    int num_partitions;
    int buffer_capacity;
    bool prefetching;
    bool conserve_memory;
};

struct TrainingOptions {
    int batch_size;
    int number_of_chunks;
    int negatives;
    float degree_fraction;
    NegativeSamplingAccess negative_sampling_access;
    float learning_rate;
    float regularization_coef;
    int regularization_norm;
    OptimizerType optimizer_type;
    LossFunctionType loss_function_type;
    float margin;
    bool average_gradients;
    bool synchronous;
    int num_epochs;
    int checkpoint_interval;
    int shuffle_interval;
};

struct TrainingPipelineOptions {
    int max_batches_in_flight;
    bool update_in_flight;
    int embeddings_host_queue_size;
    int embeddings_device_queue_size;
    int gradients_host_queue_size;
    int gradients_device_queue_size;
    int num_embedding_loader_threads;
    int num_embedding_transfer_threads;
    int num_compute_threads;
    int num_gradient_transfer_threads;
    int num_embedding_update_threads;
};

struct EvaluationOptions {
    int batch_size;
    int number_of_chunks;
    int negatives;
    float degree_fraction;
    NegativeSamplingAccess negative_sampling_access;
    int epochs_per_eval;
    bool synchronous;
    bool filtered_evaluation;
    int checkpoint_to_eval;
};

struct EvaluationPipelineOptions {
    int max_batches_in_flight;
    int embeddings_host_queue_size;
    int embeddings_device_queue_size;
    int num_embedding_loader_threads;
    int num_embedding_transfer_threads;
    int num_evaluate_threads;
};

struct PathOptions {
    string train_edges;
    string train_edges_partitions;
    string validation_edges;
    string validation_edges_partitions;
    string test_edges;
    string test_edges_partitions;
    string node_labels;
    string relation_labels;
    string node_ids;
    string relations_ids;
    string custom_ordering;
    string base_directory;
    string experiment_directory;
};

struct ReportingOptions {
    int logs_per_epoch;
    spdlog::level::level_enum log_level;
};

struct MariusOptions {
    GeneralOptions general{};
    ModelOptions model{};
    StorageOptions storage{};
    TrainingOptions training{};
    TrainingPipelineOptions training_pipeline{};
    EvaluationOptions evaluation{};
    EvaluationPipelineOptions evaluation_pipeline{};
    PathOptions path{};
    ReportingOptions reporting{};
};

MariusOptions parseConfig(int64_t argc, char *argv[]);

extern MariusOptions marius_options;

void logConfig();

#endif //MARIUS_CONFIG_H
