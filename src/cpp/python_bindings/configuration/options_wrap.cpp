#include "common/pybind_headers.h"
#include "configuration/options.h"

void init_options(py::module &m) {
    py::enum_<LearningTask>(m, "LearningTask")
        .value("NODE_CLASSIFICATION", LearningTask::NODE_CLASSIFICATION)
        .value("LINK_PREDICTION", LearningTask::LINK_PREDICTION)
        .value("ENCODE", LearningTask::ENCODE);

    m.def("getLearningTask", &getLearningTask, py::arg("string_val"));

    py::enum_<EdgeDecoderMethod>(m, "EdgeDecoderMethod").value("ONLY_POS", EdgeDecoderMethod::ONLY_POS).value("CORRUPT_NODE", EdgeDecoderMethod::CORRUPT_NODE);

    m.def("getEdgeDecoderMethod", &getEdgeDecoderMethod, py::arg("string_val"));

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
        .value("SOFTMAX_CE", LossFunctionType::SOFTMAX_CE)
        .value("RANKING", LossFunctionType::RANKING)
        .value("BCE_AFTER_SIGMOID", LossFunctionType::BCE_AFTER_SIGMOID)
        .value("BCE_WITH_LOGITS", LossFunctionType::BCE_WITH_LOGITS)
        .value("MSE", LossFunctionType::MSE)
        .value("SOFTPLUS", LossFunctionType::SOFTPLUS);

    m.def("getLossFunctionType", &getLossFunctionType, py::arg("string_val"));

    py::enum_<LossReduction>(m, "LossReduction").value("MEAN", LossReduction::MEAN).value("SUM", LossReduction::SUM);

    m.def("getLossReduction", &getLossReduction, py::arg("string_val"));

    py::enum_<ActivationFunction>(m, "ActivationFunction")
        .value("RELU", ActivationFunction::RELU)
        .value("SIGMOID", ActivationFunction::SIGMOID)
        .value("NONE", ActivationFunction::NONE);

    m.def("getActivationFunction", &getActivationFunction, py::arg("string_val"));

    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("ADAM", OptimizerType::ADAM)
        .value("ADAGRAD", OptimizerType::ADAGRAD)
        .value("DEFAULT", OptimizerType::DEFAULT);

    m.def("getOptimizerType", &getOptimizerType, py::arg("string_val"));

    py::enum_<ReductionLayerType>(m, "ReductionLayerType")
        .value("NONE", ReductionLayerType::NONE)
        .value("CONCAT", ReductionLayerType::CONCAT)
        .value("LINEAR", ReductionLayerType::LINEAR);

    m.def("getReductionLayerType", &getReductionLayerType, py::arg("string_val"));

    py::enum_<LayerType>(m, "LayerType")
        .value("NONE", LayerType::NONE)
        .value("EMBEDDING", LayerType::EMBEDDING)
        .value("FEATURE", LayerType::FEATURE)
        .value("GNN", LayerType::GNN)
        .value("DENSE", LayerType::DENSE)
        .value("REDUCTION", LayerType::REDUCTION);

    m.def("getLayerType", &getLayerType, py::arg("string_val"));

    py::enum_<DenseLayerType>(m, "DenseLayerType")
        .value("NONE", DenseLayerType::NONE)
        .value("LINEAR", DenseLayerType::LINEAR)
        .value("CONV", DenseLayerType::CONV);

    m.def("getDenseLayerType", &getDenseLayerType, py::arg("string_val"));

    py::enum_<GNNLayerType>(m, "GNNLayerType")
        .value("NONE", GNNLayerType::NONE)
        .value("GRAPH_SAGE", GNNLayerType::GRAPH_SAGE)
        .value("GCN", GNNLayerType::GCN)
        .value("GAT", GNNLayerType::GAT)
        .value("RGCN", GNNLayerType::RGCN);

    m.def("getGNNLayerType", &getGNNLayerType, py::arg("string_val"));

    py::enum_<GraphSageAggregator>(m, "GraphSageAggregator").value("GCN", GraphSageAggregator::GCN).value("MEAN", GraphSageAggregator::MEAN);

    m.def("getGraphSageAggregator", &getGraphSageAggregator, py::arg("string_val"));

    py::enum_<DecoderType>(m, "DecoderType")
        .value("NODE", DecoderType::NODE)
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
        .value("COMET", EdgeBucketOrdering::COMET)
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

    py::class_<InitOptions, std::shared_ptr<InitOptions>>(m, "InitOptions").def(py::init<>());

    py::class_<ConstantInitOptions, InitOptions, std::shared_ptr<ConstantInitOptions>>(m, "ConstantInitOptions")
        .def(py::init<float>(), py::arg("constant"))
        .def_readwrite("constant", &ConstantInitOptions::constant);

    py::class_<UniformInitOptions, InitOptions, std::shared_ptr<UniformInitOptions>>(m, "UniformInitOptions")
        .def(py::init<float>(), py::arg("scale_factor"))
        .def_readwrite("scale_factor", &UniformInitOptions::scale_factor);

    py::class_<NormalInitOptions, InitOptions, std::shared_ptr<NormalInitOptions>>(m, "NormalInitOptions")
        .def(py::init<float, float>(), py::arg("mean"), py::arg("std"))
        .def_readwrite("mean", &NormalInitOptions::mean)
        .def_readwrite("std", &NormalInitOptions::std);

    py::class_<LossOptions, std::shared_ptr<LossOptions>>(m, "LossOptions").def(py::init<>()).def_readwrite("loss_reduction", &LossOptions::loss_reduction);

    py::class_<RankingLossOptions, LossOptions, std::shared_ptr<RankingLossOptions>>(m, "RankingLossOptions")
        .def(py::init<>())
        .def_readwrite("loss_reduction", &RankingLossOptions::loss_reduction)
        .def_readwrite("margin", &RankingLossOptions::margin);

    py::class_<OptimizerOptions, std::shared_ptr<OptimizerOptions>>(m, "OptimizerOptions")
        .def(py::init<>())
        .def_readwrite("learning_rate", &OptimizerOptions::learning_rate);

    py::class_<AdagradOptions, OptimizerOptions, std::shared_ptr<AdagradOptions>>(m, "AdagradOptions")
        .def(py::init<>())
        .def_readwrite("eps", &AdagradOptions::eps)
        .def_readwrite("init_value", &AdagradOptions::init_value)
        .def_readwrite("lr_decay", &AdagradOptions::lr_decay)
        .def_readwrite("weight_decay", &AdagradOptions::weight_decay);

    py::class_<AdamOptions, OptimizerOptions, std::shared_ptr<AdamOptions>>(m, "AdamOptions")
        .def(py::init<>())
        .def_readwrite("amsgrad", &AdamOptions::amsgrad)
        .def_readwrite("beta_1", &AdamOptions::beta_1)
        .def_readwrite("beta_2", &AdamOptions::beta_2)
        .def_readwrite("eps", &AdamOptions::eps)
        .def_readwrite("weight_decay", &AdamOptions::weight_decay);

    py::class_<LayerOptions, std::shared_ptr<LayerOptions>>(m, "LayerOptions").def(py::init<>());

    py::class_<EmbeddingLayerOptions, LayerOptions, std::shared_ptr<EmbeddingLayerOptions>>(m, "EmbeddingLayerOptions").def(py::init<>());

    py::class_<FeatureLayerOptions, LayerOptions, std::shared_ptr<FeatureLayerOptions>>(m, "FeatureLayerOptions").def(py::init<>());

    py::class_<DenseLayerOptions, LayerOptions, std::shared_ptr<DenseLayerOptions>>(m, "DenseLayerOptions")
        .def(py::init<>())
        .def_readwrite("type", &DenseLayerOptions::type);

    py::class_<ReductionLayerOptions, LayerOptions, std::shared_ptr<ReductionLayerOptions>>(m, "ReductionLayerOptions")
        .def(py::init<>())
        .def_readwrite("type", &ReductionLayerOptions::type);

    py::class_<GNNLayerOptions, LayerOptions, std::shared_ptr<GNNLayerOptions>>(m, "GNNLayerOptions")
        .def(py::init<>())
        .def_readwrite("type", &GNNLayerOptions::type);

    py::class_<GraphSageLayerOptions, GNNLayerOptions, std::shared_ptr<GraphSageLayerOptions>>(m, "GraphSageLayerOptions")
        .def(py::init<>())
        .def_readwrite("aggregator", &GraphSageLayerOptions::aggregator);

    py::class_<GATLayerOptions, GNNLayerOptions, std::shared_ptr<GATLayerOptions>>(m, "GATLayerOptions")
        .def(py::init<>())
        .def_readwrite("num_heads", &GATLayerOptions::num_heads)
        .def_readwrite("average_heads", &GATLayerOptions::average_heads)
        .def_readwrite("negative_slope", &GATLayerOptions::negative_slope)
        .def_readwrite("input_dropout", &GATLayerOptions::input_dropout)
        .def_readwrite("attention_dropout", &GATLayerOptions::attention_dropout);

    py::class_<DecoderOptions, std::shared_ptr<DecoderOptions>>(m, "DecoderOptions").def(py::init<>());

    py::class_<EdgeDecoderOptions, DecoderOptions, std::shared_ptr<EdgeDecoderOptions>>(m, "EdgeDecoderOptions")
        .def(py::init<>())
        .def_readwrite("inverse_edges", &EdgeDecoderOptions::inverse_edges)
        .def_readwrite("mode", &EdgeDecoderOptions::edge_decoder_method)
        .def_readwrite("input_dim", &EdgeDecoderOptions::input_dim);

    py::class_<StorageOptions, std::shared_ptr<StorageOptions>>(m, "StorageOptions").def(py::init<>()).def_readwrite("dtype", &StorageOptions::dtype);

    py::class_<PartitionBufferOptions, StorageOptions, std::shared_ptr<PartitionBufferOptions>>(m, "PartitionBufferOptions")
        .def(py::init<>())
        .def_readwrite("num_partitions", &PartitionBufferOptions::num_partitions)
        .def_readwrite("buffer_capacity", &PartitionBufferOptions::buffer_capacity)
        .def_readwrite("prefetching", &PartitionBufferOptions::prefetching)
        .def_readwrite("fine_to_coarse_ratio", &PartitionBufferOptions::fine_to_coarse_ratio)
        .def_readwrite("edge_bucket_ordering", &PartitionBufferOptions::edge_bucket_ordering)
        .def_readwrite("node_partition_ordering", &PartitionBufferOptions::node_partition_ordering);

    py::class_<NeighborSamplingOptions, std::shared_ptr<NeighborSamplingOptions>>(m, "NeighborSamplingOptions").def(py::init<>());

    py::class_<UniformSamplingOptions, NeighborSamplingOptions, std::shared_ptr<UniformSamplingOptions>>(m, "UniformSamplingOptions")
        .def(py::init<>())
        .def_readwrite("max_neighbors", &UniformSamplingOptions::max_neighbors);

    py::class_<DropoutSamplingOptions, NeighborSamplingOptions, std::shared_ptr<DropoutSamplingOptions>>(m, "DropoutSamplingOptions")
        .def(py::init<>())
        .def_readwrite("rate", &DropoutSamplingOptions::rate);
}