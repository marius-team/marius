//
// Created by Jason Mohoney on 1/21/22.
//

#include "pipeline/pipeline_gpu.h"

#include "reporting/logger.h"
#include "pipeline/queue.h"

void BatchToDeviceWorker::run() {

    unsigned int rand_seed = rand();

    int assign_id = 0;

    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->loaded_batches_->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }
            *status_ = ThreadStatus::Running;

            int queue_choice = pipeline_->assign_id_++ % ((PipelineGPU *) pipeline_)->device_loaded_batches_.size();

            batch->to(pipeline_->model_->device_models_[queue_choice]->device_);

            *status_ = ThreadStatus::WaitPush;

            ((PipelineGPU *) pipeline_)->device_loaded_batches_[queue_choice]->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void ComputeWorkerGPU::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->device_loaded_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            pipeline_->dataloader_->loadGPUParameters(batch);

            if (pipeline_->isTrain()) {
                bool will_sync = false;
                if (pipeline_->model_->device_models_.size() > 1) {
                    ((PipelineGPU *) pipeline_)->gpu_sync_lock_->lock();
                    ((PipelineGPU *) pipeline_)->batches_since_last_sync_++;

                    if (((PipelineGPU *) pipeline_)->batches_since_last_sync_ == ((PipelineGPU *) pipeline_)->gpu_sync_interval_) {
                        will_sync = true;
                    }

                    // only release the lock if we don't need to synchronize the GPUs
                    if (!will_sync) {
                        ((PipelineGPU *) pipeline_)->gpu_sync_lock_->unlock();
                    }
                }

                if (batch->node_embeddings_.defined()) {
                    batch->node_embeddings_.requires_grad_();
                }

                batch->dense_graph_.performMap();

                pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch, ((PipelineGPU *) pipeline_)->pipeline_options_->gpu_model_average);

                if (will_sync) {
                    // we already have the lock acquired, it is safe to sync?
                    pipeline_->model_->all_reduce();

                    ((PipelineGPU *) pipeline_)->batches_since_last_sync_ = 0;
                    ((PipelineGPU *) pipeline_)->gpu_sync_lock_->unlock();
                }

                if (!pipeline_->has_embeddings()) {
                    batch->clear();
                    pipeline_->reporter_->addResult(batch->batch_size_);
                    pipeline_->batches_in_flight_--;
                    pipeline_->dataloader_->finishedBatch();
                    pipeline_->max_batches_cv_->notify_one();
                    pipeline_->edges_processed_ += batch->batch_size_;
                } else {
                    pipeline_->dataloader_->updateEmbeddings(batch, true);
                    *status_ = ThreadStatus::WaitPush;
                    ((PipelineGPU *) pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
                }
            } else {
                pipeline_->model_->device_models_[gpu_id_]->evaluate_batch(batch);

                pipeline_->batches_in_flight_--;
                pipeline_->max_batches_cv_->notify_one();
                pipeline_->dataloader_->finishedBatch();
                batch->clear();
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void EncodeNodesWorkerGPU::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->device_loaded_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            pipeline_->dataloader_->loadGPUParameters(batch);

            batch->dense_graph_.performMap();
            torch::Tensor encoded = pipeline_->model_->device_models_[gpu_id_].get()->encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
            batch->clear();
            batch->encoded_uniques_ = encoded.contiguous();

            ((PipelineGPU *) pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void BatchToHostWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->device_update_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            batch->embeddingsToHost();
            *status_ = ThreadStatus::WaitPush;

            ((PipelineGPU *) pipeline_)->update_batches_->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

PipelineGPU::PipelineGPU(shared_ptr<DataLoader> dataloader,
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
    gpu_sync_lock_ = new std::mutex();
    batches_since_last_sync_ = 0;
    gpu_sync_interval_ = pipeline_options_->gpu_sync_interval;
    assign_id_ = 0;
    encode_only_ = encode_only;

    if (train_) {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        for (int i = 0; i < model_->device_models_.size(); i++) {
            device_loaded_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_device_queue_size));
            if (model_->has_embeddings()) {
                device_update_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_device_queue_size));
            }
        }
        if (model_->has_embeddings()) {
            update_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_host_queue_size);
        }
    }  else {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        device_loaded_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_device_queue_size));
    }

    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<std::thread>;
    }

    staleness_bound_ = pipeline_options_->staleness_bound;
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    curr_pos_ = 0;

    PipelineGPU::initialize();
}


PipelineGPU::~PipelineGPU() {

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        auto threads = pool_[i];
        for (int j = 0; j < threads->size(); j++) {
            *(*pool_status_[i])[j] = ThreadStatus::Done;
            (*threads)[j].join();
        }
    }

    delete gpu_sync_lock_;

    loaded_batches_ = nullptr;
    device_loaded_batches_ = {};

    if (train_) {
        if (model_->has_embeddings()) {
            device_update_batches_ = {};
        }

        if (model_->has_embeddings()) {
            update_batches_ = nullptr;
        }
    }
}

void PipelineGPU::addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus) {
    bool *paused;
    ThreadStatus *status;

    for (int i = 0; i < num_workers; i++) {
        for (int j = 0; j < num_gpus; j++) {
            paused = new bool(true);
            status = new ThreadStatus(ThreadStatus::Paused);
            pool_paused_[pool_id]->emplace_back(paused);
            pool_status_[pool_id]->emplace_back(status);
            pool_[pool_id]->emplace_back(initThreadOfType(worker_type, paused, status, j));
        }
    }
}

void PipelineGPU::initialize() {

    if (encode_only_) {
        addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
        addWorkersToPool(2, GPU_ENCODE_ID, 1, model_->device_models_.size()); // Only one std::thread manages GPU
        if (model_->has_embeddings()) {
            addWorkersToPool(3, D2H_TRANSFER_ID, pipeline_options_->gradient_transfer_threads, model_->device_models_.size());
            addWorkersToPool(4, NODE_WRITE_ID, pipeline_options_->gradient_update_threads);
        }
    } else {
        if (train_) {
            addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
            addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
            addWorkersToPool(2, GPU_COMPUTE_ID, 1, model_->device_models_.size()); // Only one std::thread manages GPU
            if (model_->has_embeddings()) {
                addWorkersToPool(3, D2H_TRANSFER_ID, pipeline_options_->gradient_transfer_threads, model_->device_models_.size());
                addWorkersToPool(4, UPDATE_BATCH_ID, pipeline_options_->gradient_update_threads);
            }
        } else {
            addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
            addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
            addWorkersToPool(2, GPU_COMPUTE_ID, 1, model_->device_models_.size());
        }
    }
}

void PipelineGPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    assign_id_ = 0;
    setQueueExpectingData(true);

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = false;
        }
    }
}

void PipelineGPU::stopAndFlush() {

    waitComplete();
    setQueueExpectingData(false);

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = true;
        }
    }
    max_batches_cv_->notify_all();

    SPDLOG_INFO("Pipeline flush complete");
    edges_processed_ = 0;
}

void PipelineGPU::flushQueues() {
    if (train_) {
        loaded_batches_->flush();
        for (auto d : device_loaded_batches_) {
            d->flush();
        }

        if (model_->has_embeddings()) {
            for (auto d: device_update_batches_) {
                d->flush();
            }
        }

        if (model_->has_embeddings()) {
            update_batches_->flush();
        }
    } else {
        loaded_batches_->flush();
        for (auto d : device_loaded_batches_) {
            d->flush();
        }
    }
}

void PipelineGPU::setQueueExpectingData(bool expecting_data) {
    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        loaded_batches_->cv_->notify_all();
        for (auto d : device_loaded_batches_) {
            d->expecting_data_ = expecting_data;
            d->cv_->notify_all();
        }

        if (model_->has_embeddings()) {
            for (auto d : device_update_batches_) {
                d->expecting_data_ = expecting_data;
                d->cv_->notify_all();
            }
        }

        if (model_->has_embeddings()) {
            update_batches_->expecting_data_ = expecting_data;
            update_batches_->cv_->notify_all();
        }
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
        loaded_batches_->cv_->notify_all();
        for (auto d : device_loaded_batches_) {
            d->expecting_data_ = expecting_data;
            d->cv_->notify_all();
        }
    }
}
