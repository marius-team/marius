//
// Created by Jason Mohoney on 7/30/20.
//

#include "util.h"

#include <unistd.h>

#include <iostream>

#include "logger.h"

bool has_nans(torch::Tensor values) {
    return torch::isnan(values).any().item<bool>();

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
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}

void * memset_wrapper(void *ptr, int value, int64_t num) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < num) {
        curr_bytes = num - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memset((char *) ptr + local_offset, value, curr_bytes);

        local_offset += curr_bytes;
    }

    return ptr;
}

void * memcpy_wrapper(void* dest, const void* src, int64_t count) {
    int64_t curr_bytes = 0;
    int64_t local_offset = 0;

    while (local_offset < count) {
        curr_bytes = count - local_offset;
        if (curr_bytes > 1e9) {
            curr_bytes = 1e9;
        }

        memcpy((char *) dest + local_offset, (char *) src + local_offset, curr_bytes);

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

        if (pread(fd, (char *) buf + local_offset, curr_bytes, offset + local_offset) == -1) {
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

        if (pwrite(fd, (char *) buf + local_offset, curr_bytes, offset + local_offset) == -1) {
            return -1;
        }

        local_offset += curr_bytes;
    }

    return count;
}

torch::Tensor transfer_tensor(torch::Tensor input, torch::Device device) {
    if (input.defined()) {
        auto device_options = torch::TensorOptions().device(device).pinned_memory(true);
        input = input.to(device_options, true, false);
//        input.record_stream(at::cuda::getCurrentCUDAStream(device.index()));
    }
    return input;
}
