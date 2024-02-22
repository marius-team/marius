//
// Created by Jason Mohoney on 10/8/21.
//

#include "configuration/options.h"

LearningTask getLearningTask(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "NODE_CLASSIFICATION" || string_val == "NC") {
        return LearningTask::NODE_CLASSIFICATION;
    } else if (string_val == "LINK_PREDICTION" || string_val == "LP") {
        return LearningTask::LINK_PREDICTION;
    } else {
        throw std::runtime_error("Unrecognized learning task string");
    }
}

InitDistribution getInitDistribution(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

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
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "SOFTMAX_CE") {
        return LossFunctionType::SOFTMAX_CE;
    } else if (string_val == "RANKING") {
        return LossFunctionType::RANKING;
    } else if (string_val == "CROSS_ENTROPY") {
        return LossFunctionType::CROSS_ENTROPY;
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
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "MEAN") {
        return LossReduction::MEAN;
    } else if (string_val == "SUM") {
        return LossReduction::SUM;
    } else {
        throw std::runtime_error("Unrecognized loss reduction string");
    }
}

ActivationFunction getActivationFunction(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

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
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "SGD") {
        return OptimizerType::SGD;
    } else if (string_val == "ADAM") {
        return OptimizerType::ADAM;
    } else if (string_val == "ADAGRAD") {
        return OptimizerType::ADAGRAD;
    } else if (string_val == "DEFAULT") {
        return OptimizerType::DEFAULT;
    } else {
        throw std::runtime_error("Unrecognized optimizer string");
    }
}

ReductionLayerType getReductionLayerType(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "NONE") {
        return ReductionLayerType::NONE;
    } else if (string_val == "CONCAT") {
        return ReductionLayerType::CONCAT;
    } else if (string_val == "LINEAR") {
        return ReductionLayerType::LINEAR;
    } else {
        throw std::runtime_error("Unrecognized reduction type string");
    }
}

DenseLayerType getDenseLayerType(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "NONE") {
        return DenseLayerType::NONE;
    } else if (string_val == "LINEAR") {
        return DenseLayerType::LINEAR;
    } else if (string_val == "CONV") {
        return DenseLayerType::CONV;
    } else {
        throw std::runtime_error("Unrecognized dense layer string");
    }
}

LayerType getLayerType(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "NONE") {
        return LayerType::NONE;
    } else if (string_val == "EMBEDDING") {
        return LayerType::EMBEDDING;
    } else if (string_val == "FEATURE") {
        return LayerType::FEATURE;
    } else if (string_val == "GNN") {
        return LayerType::GNN;
    } else if (string_val == "DENSE") {
        return LayerType::DENSE;
    } else if (string_val == "REDUCTION") {
        return LayerType::REDUCTION;
    } else {
        throw std::runtime_error("Unrecognized layer type string");
    }
}

GNNLayerType getGNNLayerType(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

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
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "GCN") {
        return GraphSageAggregator::GCN;
    } else if (string_val == "MEAN") {
        return GraphSageAggregator::MEAN;
    } else {
        throw std::runtime_error("Unrecognized graph sage aggregator string");
    }
}

DecoderType getDecoderType(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "NODE") {
        return DecoderType::NODE;
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

EdgeDecoderMethod getEdgeDecoderMethod(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "ONLY_POS") {
        return EdgeDecoderMethod::ONLY_POS;
    } else if (string_val == "POS_AND_NEG") {
        return EdgeDecoderMethod::POS_AND_NEG;
    } else if (string_val == "CORRUPT_NODE") {
        return EdgeDecoderMethod::CORRUPT_NODE;
    } else if (string_val == "CORRUPT_REL") {
        return EdgeDecoderMethod::CORRUPT_REL;
    } else if (string_val == "TRAIN") {
        return EdgeDecoderMethod::CORRUPT_NODE;
    } else if (string_val == "INFER") {
        return EdgeDecoderMethod::ONLY_POS;
    } else {
        throw std::runtime_error("Unrecognized edge decoder type string");
    }
}

StorageBackend getStorageBackend(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

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
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "OLD_BETA") {
        return EdgeBucketOrdering::OLD_BETA;
    } else if (string_val == "NEW_BETA") {
        return EdgeBucketOrdering::NEW_BETA;
    } else if (string_val == "ALL_BETA") {
        return EdgeBucketOrdering::ALL_BETA;
    } else if (string_val == "COMET") {
        return EdgeBucketOrdering::COMET;
    } else if (string_val == "CUSTOM") {
        return EdgeBucketOrdering::CUSTOM;
    } else {
        throw std::runtime_error("Unrecognized edge bucket ordering string");
    }
}

NodePartitionOrdering getNodePartitionOrderingEnum(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

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
    for (auto& c : string_val) c = toupper(c);

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

LocalFilterMode getLocalFilterMode(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "ALL") {
        return LocalFilterMode::ALL;
    } else if (string_val == "DEG") {
        return LocalFilterMode::DEG;
    } else {
        throw std::runtime_error("Unrecognized neighbor sampling layer string");
    }
}

torch::Dtype getDtype(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "INT" || string_val == "INT32") {
        return torch::kInt32;
    } else if (string_val == "INT64" || string_val == "LONG") {
        return torch::kInt64;
    } else if (string_val == "FLOAT" || string_val == "FLOAT32") {
        return torch::kFloat32;
    } else if (string_val == "DOUBLE" || string_val == "FLOAT64") {
        return torch::kFloat64;
    } else {
        throw std::runtime_error("Unrecognized dtype string");
    }
}

spdlog::level::level_enum getLogLevel(std::string string_val) {
    for (auto& c : string_val) c = toupper(c);

    if (string_val == "ERROR" || string_val == "E") {
        return spdlog::level::err;
    } else if (string_val == "WARN" || string_val == "W") {
        return spdlog::level::warn;
    } else if (string_val == "INFO" || string_val == "I") {
        return spdlog::level::info;
    } else if (string_val == "DEBUG" || string_val == "D") {
        return spdlog::level::debug;
    } else if (string_val == "TRACE" || string_val == "T") {
        return spdlog::level::trace;
    } else {
        throw std::runtime_error("Unrecognized log level string");
    }
}