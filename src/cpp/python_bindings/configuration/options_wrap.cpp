#include <pybind11/pybind11.h>

#include "configuration/options.h"

namespace py = pybind11;

void init_options(py::module &m) {

    py::enum_<LearningTask>(m, "LearningTask")
        .value("NODE_CLASSIFICATION", LearningTask::NODE_CLASSIFICATION)
        .value("LINK_PREDICTION", LearningTask::LINK_PREDICTION);

    m.def("getLearningTask", &getLearningTask, py::arg("string_val"));

    py::enum_<InitDistribution>(m, "InitDistribution")
        .value("ZEROS", InitDistribution::ZEROS)
        .value("ONES", InitDistribution::ONES)
        .value("CONSTANT", InitDistribution::CONSTANT)
        .value("UNIFORM", InitDistribution::UNIFORM)
        .value("NORMAL", InitDistribution::NORMAL)
        .value("GLOROT_UNIFORM", InitDistribution::GLOROT_UNIFORM)
        .value("GLOROT_NORMAL", InitDistribution::GLOROT_NORMAL);

    m.def("getInitDistribution", &getInitDistribution, py::arg("string_val"));

    py::enum_<LossFunctionType>(m, "LossFunctionType")
        .value("SOFTMAX", LossFunctionType::SOFTMAX)
        .value("RANKING", LossFunctionType::RANKING)
        .value("BCE_AFTER_SIGMOID", LossFunctionType::BCE_AFTER_SIGMOID)
        .value("BCE_WITH_LOGITS", LossFunctionType::BCE_WITH_LOGITS)
        .value("MSE", LossFunctionType::MSE)
        .value("SOFTPLUS", LossFunctionType::SOFTPLUS);

    m.def("getLossFunctionType", &getLossFunctionType, py::arg("string_val"));

    py::enum_<LossReduction>(m, "LossReduction")
        .value("MEAN", LossReduction::MEAN)
        .value("SUM", LossReduction::SUM);

    m.def("getLossReduction", &getLossReduction, py::arg("string_val"));

    py::enum_<ActivationFunction>(m, "ActivationFunction")
        .value("RELU", ActivationFunction::RELU)
        .value("SIGMOID", ActivationFunction::SIGMOID)
        .value("NONE", ActivationFunction::NONE);

    m.def("getActivationFunction", &getActivationFunction, py::arg("string_val"));

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("ADAM", OptimizerType::ADAM)
        .value("ADAGRAD", OptimizerType::ADAGRAD);

    m.def("getOptimizerType", &getOptimizerType, py::arg("string_val"));

    py::enum_<FeaturizerType>(m, "FeaturizerType")
        .value("NONE", FeaturizerType::NONE)
        .value("CONCAT", FeaturizerType::CONCAT)
        .value("SUM", FeaturizerType::SUM)
        .value("MEAN", FeaturizerType::MEAN)
        .value("LINEAR", FeaturizerType::LINEAR);

    m.def("getFeaturizerType", &getFeaturizerType, py::arg("string_val"));

    py::enum_<GNNLayerType>(m, "GNNLayerType")
        .value("NONE", GNNLayerType::NONE)
        .value("GRAPH_SAGE", GNNLayerType::GRAPH_SAGE)
        .value("GCN", GNNLayerType::GCN)
        .value("GAT", GNNLayerType::GAT)
        .value("RGCN", GNNLayerType::RGCN);

    m.def("getGNNLayerType", &getGNNLayerType, py::arg("string_val"));

    py::enum_<GraphSageAggregator>(m, "GraphSageAggregator")
        .value("GCN", GraphSageAggregator::GCN)
        .value("MEAN", GraphSageAggregator::MEAN);

    m.def("getGraphSageAggregator", &getGraphSageAggregator, py::arg("string_val"));

    py::enum_<DecoderType>(m, "DecoderType")
        .value("NONE", DecoderType::NONE)
        .value("DISTMULT", DecoderType::DISTMULT)
        .value("TRANSE", DecoderType::TRANSE)
        .value("COMPLEX", DecoderType::COMPLEX);
    
    m.def("getDecoderType", &getDecoderType, py::arg("string_val"));

    py::enum_<StorageBackend>(m, "StorageBackend")
        .value("PARTITION_BUFFER", StorageBackend::PARTITION_BUFFER)
        .value("FLAT_FILE", StorageBackend::FLAT_FILE)
        .value("HOST_MEMORY", StorageBackend::HOST_MEMORY)
        .value("DEVICE_MEMORY", StorageBackend::DEVICE_MEMORY);

    m.def("getStorageBackend", &getStorageBackend, py::arg("string_val"));

    py::enum_<EdgeBucketOrdering>(m, "EdgeBucketOrdering")
        .value("OLD_BETA", EdgeBucketOrdering::OLD_BETA)
        .value("NEW_BETA", EdgeBucketOrdering::NEW_BETA)
        .value("ALL_BETA", EdgeBucketOrdering::ALL_BETA)
        .value("TWO_LEVEL_BETA", EdgeBucketOrdering::TWO_LEVEL_BETA)
        .value("CUSTOM", EdgeBucketOrdering::CUSTOM);
    
    m.def("getEdgeBucketOrderingEnum", &getEdgeBucketOrderingEnum, py::arg("string_val"));

    py::enum_<NodePartitionOrdering>(m, "NodePartitionOrdering")
        .value("DISPERSED", NodePartitionOrdering::DISPERSED)
        .value("SEQUENTIAL", NodePartitionOrdering::SEQUENTIAL)
        .value("CUSTOM", NodePartitionOrdering::CUSTOM);

    m.def("getNodePartitionOrderingEnum", &getNodePartitionOrderingEnum, py::arg("string_val"));

    py::enum_<NeighborSamplingLayer>(m, "NeighborSamplingLayer")
        .value("ALL", NeighborSamplingLayer::ALL)
        .value("UNIFORM", NeighborSamplingLayer::UNIFORM)
        .value("DROPOUT", NeighborSamplingLayer::DROPOUT);

    m.def("getNeighborSamplingLayer", &getNeighborSamplingLayer, py::arg("string_val"));

    m.def("getDtype", &getDtype, py::arg("string_val"));

    py::class_<InitOptions>(m, "InitOptions");

    py::class_<ConstantInitOptions, InitOptions>(m, "ConstantInitOptions")
        .def_readwrite("constant", &ConstantInitOptions::constant);
    
    py::class_<UniformInitOptions, InitOptions>(m, "UniformInitOptions")
        .def_readwrite("scale_factor", &UniformInitOptions::scale_factor);

    py::class_<NormalInitOptions, InitOptions>(m, "NormalInitOptions")
        .def_readwrite("mean", &NormalInitOptions::mean)
        .def_readwrite("std", &NormalInitOptions::std);

    py::class_<LossOptions>(m, "LossOptions")
        .def_readwrite("loss_reduction", &LossOptions::loss_reduction);

    py::class_<RankingLossOptions, LossOptions>(m, "RankingLossOptions")
        .def_readwrite("loss_reduction", &RankingLossOptions::loss_reduction)
        .def_readwrite("margin", &RankingLossOptions::margin);

    py::class_<OptimizerOptions>(m, "OptimizerOptions")
        .def_readwrite("learning_rate", &OptimizerOptions::learning_rate);

    py::class_<AdagradOptions, OptimizerOptions>(m, "AdagradOptions")
        .def_readwrite("eps", &AdagradOptions::eps)
        .def_readwrite("init_value", &AdagradOptions::init_value)
        .def_readwrite("lr_decay", &AdagradOptions::lr_decay)
        .def_readwrite("weight_decay", &AdagradOptions::weight_decay);

    py::class_<AdamOptions, OptimizerOptions>(m, "AdamOptions")
        .def_readwrite("amsgrad", &AdamOptions::amsgrad)
        .def_readwrite("beta_1", &AdamOptions::beta_1)
        .def_readwrite("beta_2", &AdamOptions::beta_2)
        .def_readwrite("eps", &AdamOptions::eps)
        .def_readwrite("weight_decay", &AdamOptions::weight_decay);

    py::class_<FeaturizerOptions>(m, "FeaturizerOptions");

    py::class_<GNNLayerOptions>(m, "GNNLayerOptions")
        .def_readwrite("input_dim", &GNNLayerOptions::input_dim)
        .def_readwrite("output_dim", &GNNLayerOptions::output_dim);

    py::class_<GraphSageLayerOptions, GNNLayerOptions>(m, "GraphSageLayerOptions")
        .def_readwrite("aggregator", &GraphSageLayerOptions::aggregator);

    py::class_<GATLayerOptions, GNNLayerOptions>(m, "GATLayerOptions")
        .def_readwrite("num_heads", &GATLayerOptions::num_heads)
        .def_readwrite("average_heads", &GATLayerOptions::average_heads)
        .def_readwrite("negative_slope", &GATLayerOptions::negative_slope)
        .def_readwrite("input_dropout", &GATLayerOptions::input_dropout)
        .def_readwrite("attention_dropout", &GATLayerOptions::attention_dropout);

    py::class_<DecoderOptions>(m, "DecoderOptions")
        .def_readwrite("input_dim", &DecoderOptions::input_dim)
        .def_readwrite("inverse_edges", &DecoderOptions::inverse_edges);

    py::class_<StorageOptions>(m, "StorageOptions")
        .def_readwrite("dtype", &StorageOptions::dtype);

    py::class_<PartitionBufferOptions, StorageOptions>(m, "PartitionBufferOptions")
        .def_readwrite("num_partitions", &PartitionBufferOptions::num_partitions)
        .def_readwrite("buffer_capacity", &PartitionBufferOptions::buffer_capacity)
        .def_readwrite("prefetching", &PartitionBufferOptions::prefetching)
        .def_readwrite("fine_to_coarse_ratio", &PartitionBufferOptions::fine_to_coarse_ratio)
        .def_readwrite("edge_bucket_ordering", &PartitionBufferOptions::edge_bucket_ordering)
        .def_readwrite("node_partition_ordering", &PartitionBufferOptions::node_partition_ordering);

    py::class_<NeighborSamplingOptions>(m, "NeighborSamplingOptions");

    py::class_<UniformSamplingOptions, NeighborSamplingOptions>(m, "UniformSamplingOptions")
        .def_readwrite("max_neighbors", &UniformSamplingOptions::max_neighbors);
    
    py::class_<DropoutSamplingOptions, NeighborSamplingOptions>(m, "DropoutSamplingOptions")
        .def_readwrite("rate", &DropoutSamplingOptions::rate);
}