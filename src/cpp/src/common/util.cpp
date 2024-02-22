//
// Created by Jason Mohoney on 7/30/20.
//

#include "common/util.h"

#include <unistd.h>

#include <fstream>
#include <iostream>

#include "reporting/logger.h"

void assert_no_nans(torch::Tensor values) {
    if (torch::isnan(values).any().item<bool>()) {
        throw MariusRuntimeException("Tensor contains Nans");
    }
}

void assert_no_neg(torch::Tensor values) {
    if ((values.le(-1)).any().item<bool>()) {
        throw MariusRuntimeException("Tensor contains negative values");
    }
}

void assert_in_range(torch::Tensor values, int64_t start, int64_t end) {
    if ((values.ge(start) & values.le(end)).any().item<bool>()) {
        throw MariusRuntimeException("Tensor contains is not in range: " + std::to_string(start) + "-" + std::to_string(end));
    }
}

void process_mem_usage() {
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
            ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}

void *memset_wrapper(void *ptr, int value, int64_t num) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < num) {
        curr_bytes = num - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memset((char *)ptr + local_offset, value, curr_bytes);

        local_offset += curr_bytes;
    }

    return ptr;
}

void *memcpy_wrapper(void *dest, const void *src, int64_t count) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memcpy((char *)dest + local_offset, (char *)src + local_offset, curr_bytes);

        local_offset += curr_bytes;
    }

    return dest;
}

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pread(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        if (pwrite(fd, (char *)buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

torch::Tensor transfer_tensor(torch::Tensor input, torch::Device device, CudaStream *compute_stream, CudaStream *transfer_stream) {
    if (input.defined()) {
        if (device.is_cuda() && input.device().is_cpu()) {
            input = input.pin_memory();
        }
        input = input.to(device, false);

#ifdef MARIUS_CUDA
        if (device.is_cuda() || input.device().is_cuda()) {
            if (compute_stream != nullptr) input.record_stream(*compute_stream);
            if (transfer_stream != nullptr) input.record_stream(*transfer_stream);
        }
#endif
    }

    return input;
}

int64_t get_dtype_size_wrapper(torch::Dtype dtype_) {
    if (dtype_ == torch::kFloat64) {
        return 8;
    }
    if (dtype_ == torch::kFloat32) {
        return 4;
    }
    if (dtype_ == torch::kFloat16) {
        return 2;
    }
    if (dtype_ == torch::kInt64) {
        return 8;
    }
    if (dtype_ == torch::kInt32) {
        return 4;
    }

    SPDLOG_ERROR("Unable to determine dtype_size_ for given dtype_ {}", dtype_);
    throw std::runtime_error("");
}

std::string get_directory(std::string filename) {
    assert(!filename.empty());

    string directory;
    const size_t last_slash_idx = filename.rfind('/');
    if (std::string::npos != last_slash_idx) {
        directory = filename.substr(0, last_slash_idx);
    }

    return directory;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> map_tensors(std::vector<torch::Tensor> unmapped_tensors) {
    for (auto tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw MariusRuntimeException("Input tensors must be 1D");
        }
    }

    torch::Tensor all_ids = torch::cat(unmapped_tensors);

    auto unique_tup = torch::_unique2(all_ids, true, true, false);

    torch::Tensor map = std::get<0>(unique_tup);
    torch::Tensor mapped_all_ids = std::get<1>(unique_tup);

    std::vector<torch::Tensor> mapped_tensors;

    int64_t offset = 0;
    int64_t size;
    for (auto tensor : unmapped_tensors) {
        size = tensor.size(0);
        mapped_tensors.emplace_back(mapped_all_ids.narrow(0, offset, size));
        offset += size;
    }

    return std::forward_as_tuple(map, mapped_tensors);
}

// TODO this function uses a searchsorted to find the approriate value in the map tensor
// this can be made faster on the cpu by using an std::map to perform lookups
std::vector<torch::Tensor> apply_tensor_map(torch::Tensor map, std::vector<torch::Tensor> unmapped_tensors) {
    for (auto tensor : unmapped_tensors) {
        if (tensor.sizes().size() > 1) {
            throw MariusRuntimeException("Input tensors must be 1D");
        }
    }

    std::vector<torch::Tensor> mapped_tensors;

    for (auto tensor : unmapped_tensors) {
        mapped_tensors.emplace_back(torch::searchsorted(map, tensor));
    }

    return mapped_tensors;
}