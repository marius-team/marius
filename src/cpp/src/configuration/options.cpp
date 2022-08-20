//
// Created by Jason Mohoney on 10/8/21.
//

#include "configuration/options.h"

LearningTask getLearningTask(std::string string_val) {
    if (string_val == "NODE_CLASSIFICATION") {
        return LearningTask::NODE_CLASSIFICATION;
    } else if (string_val == "LINK_PREDICTION") {
        return LearningTask::LINK_PREDICTION;
    } else {
        throw std::runtime_error("Unrecognized learning task string");
    }
}

InitDistribution getInitDistribution(std::string string_val) {
    if (string_val == "ZEROS") {
        return InitDistribution::ZEROS;
    } else if (string_val == "ONES") {
        return InitDistribution::ONES;
    } else if (string_val == "CONSTANT") {
        return InitDistribution::CONSTANT;
    } else if (string_val == "UNIFORM") {
        return InitDistribution::UNIFORM;
    } else if (string_val == "NORMAL") {
        return InitDistribution::NORMAL;
    } else if (string_val == "GLOROT_UNIFORM") {
        return InitDistribution::GLOROT_UNIFORM;
    } else if (string_val == "GLOROT_NORMAL") {
        return InitDistribution::GLOROT_NORMAL;
    } else {
        throw std::runtime_error("Unrecognized init distribution string");
    }
}

LossFunctionType getLossFunctionType(std::string string_val) {
    if (string_val == "SOFTMAX") {
        return LossFunctionType::SOFTMAX;
    } else if (string_val == "RANKING") {
        return LossFunctionType::RANKING;
    } else if (string_val == "BCE_AFTER_SIGMOID") {
        return LossFunctionType::BCE_AFTER_SIGMOID;
    } else if (string_val == "BCE_WITH_LOGITS") {
        return LossFunctionType::BCE_WITH_LOGITS;
    } else if (string_val == "MSE") {
        return LossFunctionType::MSE;
    } else if (string_val == "SOFTPLUS") {
        return LossFunctionType::SOFTPLUS;
    } else {
        throw std::runtime_error("Unrecognized loss function type string");
    }
}

LossReduction getLossReduction(std::string string_val) {
    if (string_val == "MEAN") {
        return LossReduction::MEAN;
    } else if (string_val == "SUM") {
        return LossReduction::SUM;
    } else {
        throw std::runtime_error("Unrecognized loss reduction string");
    }
}

ActivationFunction getActivationFunction(std::string string_val) {
    if (string_val == "RELU") {
        return ActivationFunction::RELU;
    } else if (string_val == "SIGMOID") {
        return ActivationFunction::SIGMOID;
    } else if (string_val == "NONE") {
        return ActivationFunction::NONE;
    } else {
        throw std::runtime_error("Unrecognized activation function string");
    }
}

OptimizerType getOptimizerType(std::string string_val) {
    if (string_val == "SGD") {
        return OptimizerType::SGD;
    } else if (string_val == "ADAM") {
        return OptimizerType::ADAM;
    } else if (string_val == "ADAGRAD") {
        return OptimizerType::ADAGRAD;
    } else {
        throw std::runtime_error("Unrecognized optimizer string");
    }
}

FeaturizerType getFeaturizerType(std::string string_val) {
    if (string_val == "NONE") {
        return FeaturizerType::NONE;
    } else if (string_val == "CONCAT") {
        return FeaturizerType::CONCAT;
    } else if (string_val == "SUM") {
        return FeaturizerType::SUM;
    } else if (string_val == "MEAN") {
        return FeaturizerType::MEAN;
    } else if (string_val == "LINEAR") {
        return FeaturizerType::LINEAR;
    } else {
        throw std::runtime_error("Unrecognized featurizer type string");
    }
}

GNNLayerType getGNNLayerType(std::string string_val) {
    if (string_val == "NONE") {
        return GNNLayerType::NONE;
    } else if (string_val == "GRAPH_SAGE") {
        return GNNLayerType::GRAPH_SAGE;
    } else if (string_val == "GCN") {
        return GNNLayerType::GCN;
    } else if (string_val == "GAT") {
        return GNNLayerType::GAT;
    } else if (string_val == "RGCN") {
        return GNNLayerType::RGCN;
    } else {
        throw std::runtime_error("Unrecognized gnn layer type string");
    }
}

GraphSageAggregator getGraphSageAggregator(std::string string_val) {
    if (string_val == "GCN") {
        return GraphSageAggregator::GCN;
    } else if (string_val == "MEAN") {
        return GraphSageAggregator::MEAN;
    } else {
        throw std::runtime_error("Unrecognized graph sage aggregator string");
    }
}

DecoderType getDecoderType(std::string string_val) {
    if (string_val == "NONE") {
        return DecoderType::NONE;
    } else if (string_val == "DISTMULT") {
        return DecoderType::DISTMULT;
    } else if (string_val == "TRANSE") {
        return DecoderType::TRANSE;
    } else if (string_val == "COMPLEX") {
        return DecoderType::COMPLEX;
    } else {
        throw std::runtime_error("Unrecognized decoder type string");
    }
}

StorageBackend getStorageBackend(std::string string_val) {
    if (string_val == "PARTITION_BUFFER") {
        return StorageBackend::PARTITION_BUFFER;
    } else if (string_val == "FLAT_FILE") {
        return StorageBackend::FLAT_FILE;
    } else if (string_val == "HOST_MEMORY") {
        return StorageBackend::HOST_MEMORY;
    } else if (string_val == "DEVICE_MEMORY") {
        return StorageBackend::DEVICE_MEMORY;
    } else {
        throw std::runtime_error("Unrecognized storage backend string");
    }
}

EdgeBucketOrdering getEdgeBucketOrderingEnum(std::string string_val) {
    if (string_val == "OLD_BETA") {
        return EdgeBucketOrdering::OLD_BETA;
    } else if (string_val == "NEW_BETA") {
        return EdgeBucketOrdering::NEW_BETA;
    } else if (string_val == "ALL_BETA") {
        return EdgeBucketOrdering::ALL_BETA;
    } else if (string_val == "TWO_LEVEL_BETA") {
        return EdgeBucketOrdering::TWO_LEVEL_BETA;
    } else if (string_val == "CUSTOM") {
        return EdgeBucketOrdering::CUSTOM;
    } else {
        throw std::runtime_error("Unrecognized edge bucket ordering string");
    }
}

NodePartitionOrdering getNodePartitionOrderingEnum(std::string string_val) {
    if (string_val == "DISPERSED") {
        return NodePartitionOrdering::DISPERSED;
    } else if (string_val == "SEQUENTIAL") {
        return NodePartitionOrdering::SEQUENTIAL;
    } else if (string_val == "CUSTOM") {
        return NodePartitionOrdering::CUSTOM;
    } else {
        throw std::runtime_error("Unrecognized node partition ordering string");
    }
}

NeighborSamplingLayer getNeighborSamplingLayer(std::string string_val) {
    if (string_val == "ALL") {
        return NeighborSamplingLayer::ALL;
    } else if (string_val == "UNIFORM") {
        return NeighborSamplingLayer::UNIFORM;
    } else if (string_val == "DROPOUT") {
        return NeighborSamplingLayer::DROPOUT;
    } else {
        throw std::runtime_error("Unrecognized neighbor sampling layer string");
    }
}

torch::Dtype getDtype(std::string string_val) {
    if (string_val == "int" || string_val == "int32") {
        return torch::kInt32;
    } else if (string_val == "int64" || string_val == "long") {
        return torch::kInt64;
    } else if (string_val == "float" || string_val == "float32") {
        return torch::kFloat32;
    } else if (string_val == "double" || string_val == "float64") {
        return torch::kFloat64;
    } else {
        throw std::runtime_error("Unrecognized dtype string");
    }
}