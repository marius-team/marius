//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_OPTIONS_H
#define MARIUS_OPTIONS_H

#include "datatypes.h"

// ENUM values
enum class LearningTask {
    NODE_CLASSIFICATION,
    LINK_PREDICTION
};

LearningTask getLearningTask(std::string string_val);

enum class InitDistribution {
    ZEROS,
    ONES,
    CONSTANT,
    UNIFORM,
    NORMAL,
    GLOROT_UNIFORM,
    GLOROT_NORMAL
};

InitDistribution getInitDistribution(std::string string_val);

enum class LossFunctionType {
    SOFTMAX,
    RANKING,
    BCE_AFTER_SIGMOID,
    BCE_WITH_LOGITS,
    MSE,
    SOFTPLUS
};

LossFunctionType getLossFunctionType(std::string string_val);

enum class LossReduction {
    MEAN,
    SUM
};

LossReduction getLossReduction(std::string string_val);

enum class ActivationFunction {
    RELU,
    SIGMOID,
    NONE
};

ActivationFunction getActivationFunction(std::string string_val);

enum class OptimizerType {
    SGD,
    ADAM,
    ADAGRAD
};

OptimizerType getOptimizerType(std::string string_val);

enum class FeaturizerType {
    NONE,
    CONCAT,
    SUM,        // Requires feature_size = embedding_size
    MEAN,       // Requires feature_size = embedding_size
    LINEAR      // Normalizes embeddings and features and concats them, and finally applying a linear layer Wx + b
};

FeaturizerType getFeaturizerType(std::string string_val);

enum class GNNLayerType {
    NONE,
    GRAPH_SAGE,
    GCN,
    GAT,
    RGCN
};

GNNLayerType getGNNLayerType(std::string string_val);

enum class GraphSageAggregator {
    GCN,
    MEAN
};

GraphSageAggregator getGraphSageAggregator(std::string string_val);

enum class DecoderType {
    NONE,
    DISTMULT,
    TRANSE,
    COMPLEX
};

DecoderType getDecoderType(std::string string_val);

enum class StorageBackend {
    PARTITION_BUFFER,
    FLAT_FILE,
    HOST_MEMORY,
    DEVICE_MEMORY
};

StorageBackend getStorageBackend(std::string string_val);

enum class EdgeBucketOrdering {
    OLD_BETA,
    NEW_BETA,
    ALL_BETA,
    TWO_LEVEL_BETA,
    CUSTOM
};

EdgeBucketOrdering getEdgeBucketOrderingEnum(std::string string_val);

enum class NodePartitionOrdering {
    DISPERSED,
    SEQUENTIAL,
    CUSTOM
};

NodePartitionOrdering getNodePartitionOrderingEnum(std::string string_val);


enum class NeighborSamplingLayer {
    ALL,
    UNIFORM,
    DROPOUT
};

NeighborSamplingLayer getNeighborSamplingLayer(std::string string_val);

torch::Dtype getDtype(std::string string_val);

struct InitOptions {
    virtual ~InitOptions() = default;
};

struct ConstantInitOptions : InitOptions {
    int constant;
};

struct UniformInitOptions : InitOptions {
    float scale_factor;
};

struct NormalInitOptions : InitOptions {
    float mean;
    float std;
};

struct LossOptions {
    LossReduction loss_reduction;

    virtual ~LossOptions() = default;
};

struct RankingLossOptions : LossOptions {
    LossReduction loss_reduction;
    float margin;
};

struct OptimizerOptions {
    float learning_rate;

    virtual ~OptimizerOptions() = default;
};

struct AdagradOptions : OptimizerOptions {
    float eps;
    float init_value;
    float lr_decay;
    float weight_decay;
};

struct AdamOptions : OptimizerOptions {
    bool amsgrad;
    float beta_1;
    float beta_2;
    float eps;
    float weight_decay;
};

struct FeaturizerOptions {};

struct GNNLayerOptions {
    int input_dim;
    int output_dim;

    virtual ~GNNLayerOptions() = default;
};

struct GraphSageLayerOptions : GNNLayerOptions {
    GraphSageAggregator aggregator;
};

struct GATLayerOptions : GNNLayerOptions {
    int num_heads;
    bool average_heads;
    float negative_slope;
    float input_dropout;
    float attention_dropout;
};

struct DecoderOptions {
    int input_dim;
    bool inverse_edges;
};

struct StorageOptions {
    torch::Dtype dtype;
    virtual ~StorageOptions() = default;
};

struct PartitionBufferOptions : StorageOptions {
    int num_partitions;
    int buffer_capacity;
    bool prefetching;
    int fine_to_coarse_ratio;
    int num_cache_partitions;
    EdgeBucketOrdering edge_bucket_ordering;
    NodePartitionOrdering node_partition_ordering;
    bool randomly_assign_edge_buckets;
};

struct NeighborSamplingOptions {
    virtual ~NeighborSamplingOptions() = default;
};

struct UniformSamplingOptions : NeighborSamplingOptions {
    int max_neighbors;
};

struct DropoutSamplingOptions : NeighborSamplingOptions {
    float rate;
};

#endif //MARIUS_OPTIONS_H
