//
// Created by Jason Mohoney on 2/29/20.
//

#include "pipeline/pipeline.h"

#include "pipeline/pipeline_cpu.h"
#include "pipeline/pipeline_gpu.h"
#include "reporting/logger.h"

Worker::Worker(Pipeline *pipeline) {
    pipeline_ = pipeline;
    sleep_time_.tv_sec = 0;
    sleep_time_.tv_nsec = WAIT_TIME;
    paused_ = true;
    done_ = false;
}

void LoadBatchWorker::run() {
    while (!done_) {
        while (!paused_) {
            // Check that 1) the total number of batches in the pipeline does not exceed the capacity
            // And 2) that the epoch has a batch left to process
            std::unique_lock lock(*pipeline_->max_batches_lock_);
            if ((pipeline_->batches_in_flight_ < pipeline_->staleness_bound_) && pipeline_->dataloader_->hasNextBatch()) {
                pipeline_->admitted_batches_++;
                pipeline_->batches_in_flight_++;
                lock.unlock();

                shared_ptr<Batch> batch = pipeline_->dataloader_->getBatch(c10::nullopt, false, worker_id_);

                if (batch == nullptr) {
                    break;
                }

                if (pipeline_->model_->device_.is_cuda()) {
                    ((PipelineGPU *)pipeline_)->loaded_batches_->blocking_push(batch);
                } else {
                    ((PipelineCPU *)pipeline_)->loaded_batches_->blocking_push(batch);
                }
            } else {
                // wait until we can try to grab a batch again
                pipeline_->max_batches_cv_->wait(lock);
                lock.unlock();
            }
        }
        nanosleep(&sleep_time_, NULL);  // wait until std::thread is not paused
    }
}

void UpdateBatchWorker::run() {
    while (!done_) {
        while (!paused_) {
            auto tup = ((PipelineGPU *)pipeline_)->update_batches_->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);

            if (!popped) {
                break;
            }

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
        nanosleep(&sleep_time_, NULL);
    }
}

void WriteNodesWorker::run() {
    while (!done_) {
        while (!paused_) {
            shared_ptr<Batch> batch;
            bool popped = false;
            if (pipeline_->model_->device_.is_cuda()) {
                auto tup = ((PipelineGPU *)pipeline_)->update_batches_->blocking_pop();
                popped = std::get<0>(tup);
                batch = std::get<1>(tup);
            } else {
                auto tup = ((PipelineCPU *)pipeline_)->update_batches_->blocking_pop();
                popped = std::get<0>(tup);
                batch = std::get<1>(tup);
            }

            if (!popped) {
                break;
            }

            pipeline_->dataloader_->graph_storage_->updatePutEncodedNodesRange(batch->start_idx_, batch->batch_size_, batch->encoded_uniques_);
            pipeline_->reporter_->addResult(batch->batch_size_);
            pipeline_->batches_in_flight_--;
            pipeline_->dataloader_->finishedBatch();
            pipeline_->max_batches_cv_->notify_one();
            pipeline_->edges_processed_ += batch->batch_size_;

            SPDLOG_TRACE("Completed: {}", batch->batch_id_);
        }
        nanosleep(&sleep_time_, NULL);
    }
}

Pipeline::~Pipeline() {
    delete max_batches_cv_;
    delete max_batches_lock_;
    delete pipeline_lock_;
}

shared_ptr<Worker> Pipeline::initWorkerOfType(int worker_type, int gpu_id, int worker_id) {
    shared_ptr<Worker> worker;

    if (worker_type == LOAD_BATCH_ID) {
        worker = std::make_shared<LoadBatchWorker>(this, worker_id);
    } else if (worker_type == H2D_TRANSFER_ID) {
        worker = std::make_shared<BatchToDeviceWorker>(this);
    } else if (worker_type == CPU_COMPUTE_ID) {
        worker = std::make_shared<ComputeWorkerCPU>(this);
    } else if (worker_type == GPU_COMPUTE_ID) {
        worker = std::make_shared<ComputeWorkerGPU>(this, gpu_id);
    } else if (worker_type == D2H_TRANSFER_ID) {
        worker = std::make_shared<BatchToHostWorker>(this, gpu_id);
    } else if (worker_type == UPDATE_BATCH_ID) {
        worker = std::make_shared<UpdateBatchWorker>(this);
    } else if (worker_type == CPU_ENCODE_ID) {
        worker = std::make_shared<EncodeNodesWorkerCPU>(this);
    } else if (worker_type == GPU_ENCODE_ID) {
        worker = std::make_shared<EncodeNodesWorkerGPU>(this, gpu_id);
    } else if (worker_type == NODE_WRITE_ID) {
        worker = std::make_shared<WriteNodesWorker>(this);
    }

    worker->spawn();
    return worker;
}
