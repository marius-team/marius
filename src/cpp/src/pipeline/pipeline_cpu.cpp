//
// Created by Jason Mohoney on 1/21/22.
//

#include "pipeline/pipeline_cpu.h"

#include "reporting/logger.h"
#include "pipeline/queue.h"

void AccumulateGradientsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            shared_ptr<Queue<shared_ptr<Batch>>> pop_queue = ((PipelineCPU *) pipeline_)->unaccumulated_batches_;
            *status_ = ThreadStatus::WaitPop;
            auto tup = pop_queue->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;

            // # TODO deprecate this worker
//            batch->accumulateGradients(pipeline_->model_->sparse_lr_);

            shared_ptr<Queue<shared_ptr<Batch>>> push_queue = ((PipelineCPU *) pipeline_)->update_batches_;
            *status_ = ThreadStatus::WaitPush;
            push_queue->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void ComputeWorkerCPU::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            shared_ptr<Queue<shared_ptr<Batch>>> pop_queue = ((PipelineCPU *) pipeline_)->loaded_batches_;
            auto tup = pop_queue->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            if (pipeline_->isTrain()) {

                if (batch->node_embeddings_.defined()) {
                    batch->node_embeddings_.requires_grad_();
                }

                batch->dense_graph_.performMap();

                pipeline_->model_->train_batch(batch);
                batch->status_ = BatchStatus::ComputedGradients;
                shared_ptr<Queue<shared_ptr<Batch>>> push_queue = ((PipelineCPU *) pipeline_)->unaccumulated_batches_;
                *status_ = ThreadStatus::WaitPush;
                push_queue->blocking_push(batch);
            } else {
                batch->dense_graph_.performMap();
                pipeline_->model_->evaluate_batch(batch);
                pipeline_->batches_in_flight_--;
                pipeline_->dataloader_->finishedBatch();
                pipeline_->max_batches_cv_->notify_one();
                batch->clear();
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void EncodeNodesWorkerCPU::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            shared_ptr<Queue<shared_ptr<Batch>>> pop_queue = ((PipelineCPU *) pipeline_)->loaded_batches_;
            auto tup = pop_queue->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;

            batch->dense_graph_.performMap();
            torch::Tensor encoded = pipeline_->model_->encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
            batch->clear();
            batch->encoded_uniques_ = encoded.contiguous();

            shared_ptr<Queue<shared_ptr<Batch>>> push_queue = ((PipelineCPU *) pipeline_)->update_batches_;
            *status_ = ThreadStatus::WaitPush;
            push_queue->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

PipelineCPU::PipelineCPU(shared_ptr<DataLoader> dataloader,
                         shared_ptr<Model> model,
                         bool train,
                         shared_ptr<ProgressReporter> reporter,
                         shared_ptr<PipelineConfig> pipeline_config,
                         bool encode_only) {

    dataloader_ = dataloader;
    model_ = model;
    reporter_ = reporter;
    train_ = train;
    edges_processed_ = 0;
    pipeline_options_ = pipeline_config;
    assign_id_ = 0;
    encode_only_ = encode_only;

    if (train_) {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        unaccumulated_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        update_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_host_queue_size);
    } else {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
    }

    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<std::thread>;
    }

    staleness_bound_ = pipeline_options_->staleness_bound;
    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    curr_pos_ = 0;

    PipelineCPU::initialize();
}

PipelineCPU::~PipelineCPU() {

    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        auto threads = pool_[i];
        for (int j = 0; j < threads->size(); j++) {
            *(*pool_status_[i])[j] = ThreadStatus::Done;
            (*threads)[j].join();
        }
    }

    if (train_) {
        loaded_batches_ = nullptr;
        unaccumulated_batches_ = nullptr;
        update_batches_  = nullptr;
    } else {
        loaded_batches_  = nullptr;
    }
}

bool Pipeline::isDone() {
    return (batches_in_flight_ <= 0) && dataloader_->epochComplete();
}

bool Pipeline::isTrain() {
    return train_;
}

bool Pipeline::has_embeddings() {
    return model_->has_embeddings();
}

void Pipeline::waitComplete() {
    timespec sleep_time{};
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = MILLISECOND; // check every 1 millisecond
    while (!isDone()) {
        nanosleep(&sleep_time, NULL);
    }
}

void PipelineCPU::addWorkersToPool(int pool_id, int worker_type, int num_workers, int gpu_id) {
    bool *paused;
    ThreadStatus *status;

    for (int i = 0; i < num_workers; i++) {
        paused = new bool(true);
        status = new ThreadStatus(ThreadStatus::Paused);
        pool_paused_[pool_id]->emplace_back(paused);
        pool_status_[pool_id]->emplace_back(status);
        pool_[pool_id]->emplace_back(initThreadOfType(worker_type, paused, status));
    }
}

void PipelineCPU::initialize() {

    if (encode_only_) {
        addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, CPU_ENCODE_ID, pipeline_options_->compute_threads);
        addWorkersToPool(2, NODE_WRITE_ID, pipeline_options_->gradient_update_threads);
    } else {
        if (train_) {
            addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
            addWorkersToPool(1, CPU_COMPUTE_ID, pipeline_options_->compute_threads);
            addWorkersToPool(2, CPU_ACCUMULATE_ID, pipeline_options_->gradient_update_threads);
            addWorkersToPool(3, UPDATE_BATCH_ID, pipeline_options_->gradient_update_threads);
        } else {
            addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
            addWorkersToPool(1, CPU_COMPUTE_ID, pipeline_options_->compute_threads);
        }
    }
}

void PipelineCPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    assign_id_ = 0;
    setQueueExpectingData(true);
    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = false;
        }
    }
}

void PipelineCPU::stopAndFlush() {

    waitComplete();
    setQueueExpectingData(false);

    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = true;
        }
    }

    max_batches_cv_->notify_all();
    SPDLOG_INFO("Pipeline flush complete");
    edges_processed_ = 0;
}

void PipelineCPU::flushQueues() {
    if (train_) {
        loaded_batches_->flush();
        unaccumulated_batches_->flush();
        update_batches_->flush();
    } else {
        loaded_batches_->flush();
    }
}

void PipelineCPU::setQueueExpectingData(bool expecting_data) {
    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        loaded_batches_->cv_->notify_all();
        unaccumulated_batches_->expecting_data_ = expecting_data;
        unaccumulated_batches_->cv_->notify_all();
        update_batches_->expecting_data_ = expecting_data;
        update_batches_->cv_->notify_all();
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
        loaded_batches_->cv_->notify_all();
    }
}