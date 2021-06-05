#include <pybind11/pybind11.h>

#include "datatypes.h"

namespace py = pybind11;

void init_datatypes(py::module &m) {
	// InitializationDistribution enum
    py::enum_<InitializationDistribution>(m, "InitializationDistribution")
        .value("Uniform", InitializationDistribution::Uniform)
        .value("Normal", InitializationDistribution::Normal);

    // NegativeSamplingAccess enum
    py::enum_<NegativeSamplingAccess>(m, "NegativeSamplingAccess")
        .value("Uniform", NegativeSamplingAccess::Uniform)
        .value("UniformCrossPartition", NegativeSamplingAccess::UniformCrossPartition)
        .value("All", NegativeSamplingAccess::All);

    // GraphOrdering enum
    py::enum_<EdgeBucketOrdering>(m, "EdgeBucketOrdering")
        .value("Random", EdgeBucketOrdering::Random)
        .value("RandomSymmetric", EdgeBucketOrdering::RandomSymmetric)
        .value("Sequential", EdgeBucketOrdering::Sequential)
        .value("SequentialSymmetric", EdgeBucketOrdering::SequentialSymmetric)
        .value("Hilbert", EdgeBucketOrdering::Hilbert)
        .value("HilbertSymmetric", EdgeBucketOrdering::HilbertSymmetric)
        .value("Elimination", EdgeBucketOrdering::Elimination)
        .value("Custom", EdgeBucketOrdering::Custom);

    py::enum_<EncoderModelType>(m, "EncoderModelType")
        .value("None", EncoderModelType::None)
        .value("Custom", EncoderModelType::Custom);

    py::enum_<DecoderModelType>(m, "DecoderModelType")
        .value("NodeClassification", DecoderModelType::NodeClassification)
        .value("DistMult", DecoderModelType::DistMult)
        .value("ComplEx", DecoderModelType::ComplEx)
        .value("TransE", DecoderModelType::TransE)
        .value("Custom", DecoderModelType::Custom);


    // BackendType enum
    py::enum_<BackendType>(m, "BackendType")
        .value("RocksDB", BackendType::RocksDB)
        .value("PartitionBuffer", BackendType::PartitionBuffer)
        .value("FlatFile", BackendType::FlatFile)
        .value("HostMemory", BackendType::HostMemory)
        .value("DeviceMemory", BackendType::DeviceMemory);

     // OptimizerType enum
    py::enum_<OptimizerType>(m, "OptimizerType")
        .value("SGD", OptimizerType::SGD)
        .value("Adagrad", OptimizerType::Adagrad);

    // ComparatorType enum
    py::enum_<ComparatorType>(m, "ComparatorType")
        .value("Dot", ComparatorType::Dot)
        .value("Cosine", ComparatorType::Cosine);

    // LossFunctionType enum
    py::enum_<LossFunctionType>(m, "LossFunctionType")
        .value("SoftMax", LossFunctionType::SoftMax)
        .value("RankingLoss", LossFunctionType::RankingLoss);

    // RelationOperatorType enum
    py::enum_<RelationOperatorType>(m, "RelationOperatorType")
        .value("Translation", RelationOperatorType::Translation)
        .value("ComplexHadamard", RelationOperatorType::ComplexHadamard)
        .value("Hadamard", RelationOperatorType::Hadamard)
        .value("NoOp", RelationOperatorType::NoOp);
}
