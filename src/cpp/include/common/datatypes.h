//
// Created by jasonmohoney on 10/19/19.
//

#ifndef MARIUS_DATATYPES_H
#define MARIUS_DATATYPES_H

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/exception.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop

using std::map;
using std::shared_ptr;
using std::string;
using std::unique_ptr;

/** Program Constants */

#define MAX_READ_SIZE 1E8

/** Deployment configs */

// Dummy CUDA objects so we don't break the CPU-only build
class DummyCudaEvent {
   public:
    DummyCudaEvent(int val) { (void)val; }

    void start(){};

    void record(){};

    void synchronize(){};

    int elapsed_time(DummyCudaEvent) { return 0; }
};

class DummyCudaStream {
   public:
    DummyCudaStream() {}

    void synchronize(){};
};

class DummyCudaStreamGuard {
   public:
    DummyCudaStreamGuard(DummyCudaStream) {}
};

#ifdef MARIUS_CUDA
    #include <ATen/cuda/CUDAContext.h>
    #include <ATen/cuda/CUDAEvent.h>
    #include <ATen/cuda/Exceptions.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAStream.h>
    #include <c10/util/Exception.h>

typedef at::cuda::CUDAEvent CudaEvent;
typedef at::cuda::CUDAStream CudaStream;
typedef at::cuda::CUDAStreamGuard CudaStreamGuard;

using at::cuda::getStreamFromPool;

#else
typedef DummyCudaEvent CudaEvent;
typedef DummyCudaStream CudaStream;
typedef DummyCudaStreamGuard CudaStreamGuard;

inline CudaStream getStreamFromPool(bool = false, int = 0) { return CudaStream(); }
#endif

#ifndef IO_FLAGS
    #define IO_FLAGS 0
#endif

/** Typedefs */

/**
 * Tensor of edges in COO format with node and relation indices. Shape (n, 3)
 * First column -> src_idx
 * Second column -> rel_idx
 * Third column -> dst_idx
 */
typedef torch::Tensor EdgeList;

/** 1D Tensor of indices. Shape (n) */
typedef torch::Tensor Indices;

/** Tensor of gradients. Shape: (n, EMBEDDING_SIZE) */
typedef torch::Tensor Gradients;

/** Tensor containing optimizer state for a selection of parameters. Shape: (n, FEATURE_SIZE) */
typedef torch::Tensor OptimizerState;

typedef std::chrono::time_point<std::chrono::steady_clock> Timestamp;

#endif  // MARIUS_DATATYPES_H
