//
// Created by Jason Mohoney on 7/30/20.
//

#ifndef MARIUS_UTIL_H
#define MARIUS_UTIL_H

#include "datatypes.h"

class Timer {
  public:
    bool gpu_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
    CudaEvent *start_event_;
    CudaEvent *end_event_;

    Timer(bool gpu) {
        start_event_ = new CudaEvent(0);
        end_event_ = new CudaEvent(0);
        gpu_ = gpu;
    }

    ~Timer() {
        delete start_event_;
        delete end_event_;
    }

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        if (gpu_) {
            start_event_->record();
        }
    }

    void stop() {
        stop_time_ = std::chrono::high_resolution_clock::now();
        if (gpu_) {
            end_event_->record();
        }
    }

    int64_t getDuration() {
        int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_).count();
        if (gpu_) {
            start_event_->synchronize();
            end_event_->synchronize();
            duration = start_event_->elapsed_time(*end_event_);
        }
        return duration;
    }
};

bool has_nans(torch::Tensor values);

void process_mem_usage();

void * memset_wrapper(void *ptr, int value, int64_t num);

void * memcpy_wrapper(void* dest, const void* src, int64_t count);

int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset);

int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset);

torch::Tensor transfer_tensor(torch::Tensor input, torch::Device device);

#endif //MARIUS_UTIL_H
