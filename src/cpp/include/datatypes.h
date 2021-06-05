//
// Created by jasonmohoney on 10/19/19.
//
#ifndef MARIUS_DATATYPES_H
#define MARIUS_DATATYPES_H

#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <tuple>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

using std::string;
using std::map;

/** Program Constants */

#define MAX_READ_SIZE 1E8

/** Deployment configs */

class DummyCuda {
  public:
    DummyCuda(int val) {
        (void) val;
    }

    void start() {};

    void record() {};

    void synchronize() {};

    int elapsed_time(DummyCuda) {
        return 0;
    }
};

#ifdef MARIUS_CUDA
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDAStream.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <ATen/cuda/Exceptions.h>
    #include <c10/util/Exception.h>
    #include <ATen/cuda/CUDAEvent.h>
    typedef at::cuda::CUDAEvent CudaEvent;
#else
typedef DummyCuda CudaEvent;
#endif

// TODO enable direct IO support
//#if __linux__
//    #define IO_FLAGS O_DIRECT
//#endif

#ifndef IO_FLAGS
    #define IO_FLAGS 0
#endif

/** Typedefs */

// Tensor of edges with node and relation indices. Shape (n, 3)
// First column -> src_idx
// Second column -> rel_idx
// Third column -> dst_idx
typedef torch::Tensor EdgeList;

// Tensor of embedding vectors. Shape: (n, FEATURE_SIZE)
typedef torch::Tensor Embeddings;

// Single embedding vector. Shape (FEATURE_SIZE)
typedef torch::Tensor Embedding;

// Tensor of relation vectors. Shape: (n, FEATURE_SIZE)
typedef torch::Tensor Relations;

// Single relation vector. Shape (FEATURE_SIZE)
typedef torch::Tensor Relation;

// 1D Tensor of indices. Shape (n)
typedef torch::Tensor Indices;

// Tensor of gradients. Shape: (n, FEATURE_SIZE)
typedef torch::Tensor Gradients;

// Tensor containing optimizer state for a selection of parameters. Shape: (n, FEATURE_SIZE)
typedef torch::Tensor OptimizerState;

// Sparse Tensor
typedef torch::Tensor SparseAdjacencyMatrix;

// Sparse Tensor for multiple relations (n x n x r)
typedef torch::Tensor SparseAdjacencyMatrixMR;

typedef std::chrono::time_point<std::chrono::steady_clock> Timestamp;


/** Enums */
enum class InitializationDistribution {
    Uniform,               // Initialize with torch::uniform with min: -scale_factor and max: scale_factor
    Normal                 // Initialize with torch::randn with mean 0 and variance scale_factor^2
};

enum class NegativeSamplingAccess {
    Uniform,               // Uniformly random sampling
    UniformCrossPartition, // Used with partitioning, samples come from across partitions in the buffer
    All
};

enum class EdgeBucketOrdering {
    Random,
    RandomSymmetric,
    Sequential,
    SequentialSymmetric,
    Hilbert,
    HilbertSymmetric,
    Elimination,
    Custom
};

// Currently unsupported
enum class EncoderModelType {
    None,
    Custom
};

enum class DecoderModelType {
    NodeClassification,
    DistMult,
    ComplEx,
    TransE,
    Custom
};

enum class BackendType {
    RocksDB,
    PartitionBuffer,
    FlatFile,
    HostMemory,
    DeviceMemory
};

// Optimizer options
enum class OptimizerType {
    SGD,            // Standard SGD optimizer
    Adagrad         // Standard Adagrad optimizer with state managed on disk
};

// Comparison operator options
enum class ComparatorType {
    Dot,            // Dot Product Comparator
    Cosine          // Cosine Product Comparator
};

enum class LossFunctionType {
    SoftMax,            // Dot Product Comparator
    RankingLoss,          // Cosine Product Comparator
    BCEAfterSigmoidLoss,  // Binary cross entropy after explicit Sigmoid
    BCEWithLogitsLoss,    // Combination of Sigmoid and BCELoss
    MSELoss,             // Mean square error loss
    SoftPlusLoss        // Softplus loss
};

enum class ReductionType {
    Mean,
    Sum
};

enum class RelationOperatorType {
    Translation,
    ComplexHadamard,
    Hadamard,
    NoOp
};
#endif //MARIUS_DATATYPES_H
