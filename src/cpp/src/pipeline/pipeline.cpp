//
// Created by Jason Mohoney on 2/29/20.
//

#include "pipeline/pipeline.h"

#include "pipeline/pipeline_cpu.h"
#include "pipeline/pipeline_gpu.h"
#include "reporting/logger.h"

Worker::Worker(Pipeline *pipeline, bool *paused, ThreadStatus *status) {
    pipeline_ = pipeline;
    sleep_time_.tv_sec = 0;
    sleep_time_.tv_nsec = WAIT_TIME;
    paused_ = paused;
    status_ = status;
}

void LoadBatchWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;

            // Check that 1) the total number of batches in the pipeline does not exceed the capacity
            // And 2) that the epoch has a batch left to process
            std::unique_lock lock(*pipeline_->max_batches_lock_);
            if ((pipeline_->batches_in_flight_ < pipeline_->staleness_bound_) && pipeline_->dataloader_->hasNextBatch()) {
                pipeline_->admitted_batches_++;
                pipeline_->batches_in_flight_++;
//                lock.unlock();

                *status_ = ThreadStatus::Running;
                shared_ptr<Batch> batch = pipeline_->dataloader_->getBatch();
                lock.unlock(); // TODO make sure having the unlock after getBatch doesn't introduce deadlock

                if (batch == nullptr) {
                    break;
                }

                *status_ = ThreadStatus::WaitPush;
                if (pipeline_->model_->device_.is_cuda()) {
                    ((PipelineGPU *) pipeline_)->loaded_batches_->blocking_push(batch);
                } else {
                    ((PipelineCPU *) pipeline_)->loaded_batches_->blocking_push(batch);
                }
            } else {
                // wait until we can try to grab a batch again
                pipeline_->max_batches_cv_->wait(lock);
                lock.unlock();
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL); // wait until std::thread is not paused
    }
}

void UpdateBatchWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            auto tup = ((PipelineGPU *) pipeline_)->update_batches_->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);

            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;


            // transfer gradients and update parameters
            if (batch->node_embeddings_.defined()) {
                pipeline_->dataloader_->updateEmbeddings(batch, false);
            }

            pipeline_->reporter_->addResult(batch->batch_size_);
            pipeline_->batches_in_flight_--;
            pipeline_->dataloader_->finishedBatch();
            pipeline_->max_batches_cv_->notify_one();
            pipeline_->edges_processed_ += batch->batch_size_;

            SPDLOG_TRACE("Completed: {}", batch->batch_id_);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void WriteNodesWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {

            shared_ptr<Batch> batch;
            bool popped = false;
            if (pipeline_->model_->device_.is_cuda()) {
                auto tup = ((PipelineGPU *) pipeline_)->update_batches_->blocking_pop();
                popped = std::get<0>(tup);
                batch = std::get<1>(tup);
            } else {
                auto tup = ((PipelineCPU *) pipeline_)->update_batches_->blocking_pop();
                popped = std::get<0>(tup);
                batch = std::get<1>(tup);
            }

            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            pipeline_->dataloader_->graph_storage_->updatePutEncodedNodesRange(batch->start_idx_, batch->batch_size_, batch->encoded_uniques_);
            pipeline_->reporter_->addResult(batch->batch_size_);
            pipeline_->batches_in_flight_--;
            pipeline_->dataloader_->finishedBatch();
            pipeline_->max_batches_cv_->notify_one();
            pipeline_->edges_processed_ += batch->batch_size_;

            SPDLOG_TRACE("Completed: {}", batch->batch_id_);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

Pipeline::~Pipeline() {
    delete max_batches_cv_;
    delete max_batches_lock_;
    delete pipeline_lock_;
}

std::thread Pipeline::initThreadOfType(int worker_type, bool *paused, ThreadStatus *status, int gpu_id) {
    std::thread t;

    if (worker_type == LOAD_BATCH_ID) {
        auto load_embeddings_func = [](LoadBatchWorker w) { w.run(); };
        t = std::thread(load_embeddings_func, LoadBatchWorker(this, paused, status));
    } else if (worker_type == H2D_TRANSFER_ID) {
        auto embeddings_to_device_func = [](BatchToDeviceWorker w) { w.run(); };
        t = std::thread(embeddings_to_device_func, BatchToDeviceWorker(this, paused, status));
    } else if (worker_type == CPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerCPU w) { w.run(); };
        t = std::thread(compute_func, ComputeWorkerCPU(this, paused, status));
    } else if (worker_type == GPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerGPU w) { w.run(); };
        t = std::thread(compute_func, ComputeWorkerGPU(this, paused, status, gpu_id));
    } else if (worker_type == CPU_ACCUMULATE_ID) {
        auto compute_func = [](AccumulateGradientsWorker w) { w.run(); };
        t = std::thread(compute_func, AccumulateGradientsWorker(this, paused, status));
    } else if (worker_type == D2H_TRANSFER_ID) {
        auto gradients_to_host_func = [](BatchToHostWorker w) { w.run(); };
        t = std::thread(gradients_to_host_func, BatchToHostWorker(this, paused, status, gpu_id));
    } else if (worker_type == UPDATE_BATCH_ID) {
        auto update_embeddings_func = [](UpdateBatchWorker w) { w.run(); };
        t = std::thread(update_embeddings_func, UpdateBatchWorker(this, paused, status));
    } else if (worker_type == CPU_ENCODE_ID) {
        auto encode_func = [](EncodeNodesWorkerCPU w) { w.run(); };
        t = std::thread(encode_func, EncodeNodesWorkerCPU(this, paused, status));
    } else if (worker_type == GPU_ENCODE_ID) {
        auto encode_func = [](EncodeNodesWorkerGPU w) { w.run(); };
        t = std::thread(encode_func, EncodeNodesWorkerGPU(this, paused, status, gpu_id));
    } else if (worker_type == NODE_WRITE_ID) {
        auto write_func = [](WriteNodesWorker w) { w.run(); };
        t = std::thread(write_func, WriteNodesWorker(this, paused, status));
    }
    return t;
}

