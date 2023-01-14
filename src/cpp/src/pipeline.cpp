//
// Created by Jason Mohoney on 2/29/20.
//

#include "pipeline.h"

#include "logger.h"

template<class T>
Queue<T>::Queue(int max_size) {
    queue_ = std::deque<T>();
    max_size_ = max_size;
    mutex_ = new std::mutex();
    cv_ = new std::condition_variable();
    expecting_data_ = true;
}

Worker::Worker(Pipeline *pipeline, bool *paused, ThreadStatus *status, int worker_id) {
    pipeline_ = pipeline;
    sleep_time_.tv_sec = 0;
    sleep_time_.tv_nsec = WAIT_TIME;
    paused_ = paused;
    status_ = status;
    worker_id_ = worker_id;
}

void LoadEmbeddingsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;

            // Check that 1) the total number of batches in the pipeline does not exceed the capacity
            // And 2) that the epoch has a batch left to process
            std::unique_lock lock(*pipeline_->max_batches_lock_);
            if ((pipeline_->batches_in_flight_ < pipeline_->staleness_bound_) && pipeline_->dataloader_->hasNextBatch()) {
                pipeline_->admitted_batches_++;
                pipeline_->batches_in_flight_++;
                lock.unlock();

                *status_ = ThreadStatus::Running;

                if (pipeline_->model_->current_device_.is_cuda()) {
                    std::vector<Batch *> batches;
                    if (pipeline_->model_->devices_.size() > 1) {
                        batches = pipeline_->dataloader_->getSubBatches();
                    } else {
                        batches = {pipeline_->dataloader_->getBatch(worker_id_)};
                    }

                    if (batches[0] == nullptr) {
                        break;
                    }

                    *status_ = ThreadStatus::WaitPush;
                    ((PipelineGPU *) pipeline_)->loaded_batches_->blocking_push(batches);
                } else {
                    Batch *batch = pipeline_->dataloader_->getBatch(worker_id_);

                    if (batch == nullptr) {
                        break;
                    }

                    *status_ = ThreadStatus::WaitPush;
                    ((PipelineCPU *) pipeline_)->loaded_batches_->blocking_push(batch);
                }
            } else {
                // wait until we can try to grab a batch again
                pipeline_->max_batches_cv_->wait(lock);
                lock.unlock();
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL); // wait until thread is not paused
    }
}

void BatchToDeviceWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->loaded_batches_->blocking_pop();
            bool popped = get<0>(tup);
            std::vector<Batch *> batches = get<1>(tup);
            if (!popped) {
                break;
            }
            *status_ = ThreadStatus::Running;

            #pragma omp parallel for
            for (int i = 0; i < batches.size(); i++) {
                batches[i]->to(pipeline_->model_->devices_[i], pipeline_->dataloader_->compute_stream_);
            }

            *status_ = ThreadStatus::WaitPush;
            ((PipelineGPU *) pipeline_)->device_loaded_batches_->blocking_push(batches);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }

}

void ComputeWorkerCPU::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            Queue<Batch *> *pop_queue = ((PipelineCPU *) pipeline_)->loaded_batches_;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            if (pipeline_->isTrain()) {
                pipeline_->model_->train_batch(batch);
                batch->status_ = BatchStatus::ComputedGradients;
                Queue<Batch *> *push_queue = ((PipelineCPU *) pipeline_)->unaccumulated_batches_;
                *status_ = ThreadStatus::WaitPush;
                push_queue->blocking_push(batch);
            } else {
                pipeline_->model_->evaluate(batch, pipeline_->dataloader_->graph_storage_->filtered_eval_);
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

void ComputeWorkerGPU::run() {
    at::cuda::CUDAStream compute_stream = at::cuda::getStreamFromPool(true, 0);
    pipeline_->dataloader_->compute_stream_ = &compute_stream;
    at::cuda::CUDAStreamGuard stream_guard(compute_stream);

    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->device_loaded_batches_->blocking_pop();
            bool popped = get<0>(tup);
            std::vector<Batch *> batches = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;

            #pragma omp parallel for
            for (int i = 0; i < batches.size(); i++) {
                pipeline_->dataloader_->loadGPUParameters(batches[i]);
            }

            if (pipeline_->isTrain()) {
                if (batches.size() > 1) {
                    pipeline_->model_->train_batch(batches);
                } else {
                    pipeline_->model_->train_batch(batches[0]);
                }

                // end here for node classification
                if (!pipeline_->hasEmbeddings()) {
                    #pragma omp parallel for
                    for (int i = 0; i < batches.size(); i++) {
                        batches[i]->clear();
                    }

                    Batch *batch;
                    bool alloc_batch = false;
                    if (batches.size() > 1) {
                        batch = new Batch(batches);
                        alloc_batch = true;
                    } else {
                        batch = batches[0];
                    }

                    pipeline_->reporter_->addResult(batch->batch_size_);
                    pipeline_->batches_in_flight_--;
                    pipeline_->dataloader_->finishedBatch();
                    pipeline_->max_batches_cv_->notify_one();
                    pipeline_->edges_processed_ += batch->batch_size_;
                    if (alloc_batch) {
                        delete batch;
                    }
                } else {
                    #pragma omp parallel for
                    for (int i = 0; i < batches.size(); i++) {
                        batches[i]->accumulateGradients(pipeline_->model_->model_config_->embeddings->optimizer->options->learning_rate);
                        pipeline_->dataloader_->updateEmbeddingsForBatch(batches[i], true);
                    }

                    *status_ = ThreadStatus::WaitPush;
                    ((PipelineGPU *) pipeline_)->device_update_batches_->blocking_push(batches);
                }
            } else {
                if (batches.size() > 1) {
                    pipeline_->model_->evaluate(batches, pipeline_->dataloader_->graph_storage_->filtered_eval_);
                } else {
                    pipeline_->model_->evaluate(batches[0], pipeline_->dataloader_->graph_storage_->filtered_eval_);
                }

                pipeline_->batches_in_flight_--;
                pipeline_->max_batches_cv_->notify_one();
                pipeline_->dataloader_->finishedBatch();

                #pragma omp parallel for
                for (int i = 0; i < batches.size(); i++) {
                    batches[i]->clear();
                }
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void AccumulateGradientsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            Queue<Batch *> *pop_queue = ((PipelineCPU *) pipeline_)->unaccumulated_batches_;
            *status_ = ThreadStatus::WaitPop;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;

            // ugly
            batch->accumulateGradients(pipeline_->model_->model_config_->embeddings->optimizer->options->learning_rate);

            Queue<Batch *> *push_queue = ((PipelineCPU *) pipeline_)->update_batches_;
            *status_ = ThreadStatus::WaitPush;
            push_queue->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void GradientsToHostWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            auto tup = ((PipelineGPU *) pipeline_)->device_update_batches_->blocking_pop();
            bool popped = get<0>(tup);
            std::vector<Batch *> batches = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            #pragma omp parallel for
            for (int i = 0; i < batches.size(); i++) {
                batches[i]->embeddingsToHost();
            }

            *status_ = ThreadStatus::WaitPush;
            ((PipelineGPU *) pipeline_)->update_batches_->blocking_push(batches);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void UpdateEmbeddingsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            Batch *batch;

            bool alloc_batch = false;
            if (pipeline_->model_->current_device_.is_cuda()) {
                auto tup = ((PipelineGPU *) pipeline_)->update_batches_->blocking_pop();
                bool popped = get<0>(tup);
                std::vector<Batch *> batches = get<1>(tup);

                if (!popped) {
                    break;
                }

                *status_ = ThreadStatus::Running;

                if (batches.size() > 1) {
                    batch = new Batch(batches);
                    alloc_batch = true;
                } else {
                    batch = batches[0];
                }

            } else {
                auto tup = ((PipelineCPU *) pipeline_)->update_batches_->blocking_pop();

                bool popped = get<0>(tup);
                batch = get<1>(tup);

                if (!popped) {
                    break;
                }

                *status_ = ThreadStatus::Running;
            }

            pipeline_->dataloader_->updateEmbeddingsForBatch(batch, false);
            pipeline_->reporter_->addResult(batch->batch_size_);
            pipeline_->batches_in_flight_--;
            pipeline_->dataloader_->finishedBatch();
            if (alloc_batch) {
                delete batch;
            }
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

thread Pipeline::initThreadOfType(int worker_type, bool *paused, ThreadStatus *status, int worker_id) {
    thread t;

    if (worker_type == EMBEDDINGS_LOADER_ID) {
        auto load_embeddings_func = [](LoadEmbeddingsWorker w) { w.run(); };
        t = thread(load_embeddings_func, LoadEmbeddingsWorker(this, paused, status, worker_id));
    } else if (worker_type == EMBEDDINGS_TRANSFER_ID) {
        auto embeddings_to_device_func = [](BatchToDeviceWorker w) { w.run(); };
        t = thread(embeddings_to_device_func, BatchToDeviceWorker(this, paused, status, worker_id));
    } else if (worker_type == CPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerCPU w) { w.run(); };
        t = thread(compute_func, ComputeWorkerCPU(this, paused, status, worker_id));
    } else if (worker_type == GPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerGPU w) { w.run(); };
        t = thread(compute_func, ComputeWorkerGPU(this, paused, status, worker_id));
    } else if (worker_type == CPU_ACCUMULATE_ID) {
        auto compute_func = [](AccumulateGradientsWorker w) { w.run(); };
        t = thread(compute_func, AccumulateGradientsWorker(this, paused, status, worker_id));
    } else if (worker_type == UPDATE_TRANSFER_ID) {
        auto gradients_to_host_func = [](GradientsToHostWorker w) { w.run(); };
        t = thread(gradients_to_host_func, GradientsToHostWorker(this, paused, status, worker_id));
    } else if (worker_type == UPDATE_EMBEDDINGS_ID) {
        auto update_embeddings_func = [](UpdateEmbeddingsWorker w) { w.run(); };
        t = thread(update_embeddings_func, UpdateEmbeddingsWorker(this, paused, status, worker_id));
    }
    return t;
}

PipelineCPU::PipelineCPU(DataLoader *dataloader,
                         std::shared_ptr<Model> model,
                         bool train,
                         ProgressReporter *reporter,
                         shared_ptr<PipelineConfig> pipeline_config) {

    dataloader_ = dataloader;
    model_ = model;
    reporter_ = reporter;
    train_ = train;
    edges_processed_ = 0;
    pipeline_options_ = pipeline_config;

    if (train_) {
        loaded_batches_ = new Queue<Batch *>(pipeline_options_->batch_host_queue_size);
        unaccumulated_batches_ = new Queue<Batch *>(pipeline_options_->batch_host_queue_size);
        update_batches_ = new Queue<Batch *>(pipeline_options_->gradients_host_queue_size);
    } else {
        loaded_batches_ = new Queue<Batch *>(pipeline_options_->batch_host_queue_size);
    }

    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<thread>;
    }

    staleness_bound_ = pipeline_options_->staleness_bound;
    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();
    batches_in_flight_ = 0;
    admitted_batches_ = 0;

    PipelineCPU::initialize();
}

PipelineCPU::~PipelineCPU() {
    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        delete pool_paused_[i];
        delete pool_status_[i];
        delete pool_[i];
    }

    if (train_) {
        delete loaded_batches_;
        delete unaccumulated_batches_;
        delete update_batches_;
    } else {
        delete loaded_batches_;
    }
}

bool Pipeline::isDone() {
    return (batches_in_flight_ <= 0) && dataloader_->epochComplete();
}

bool Pipeline::isTrain() {
    return train_;
}

bool Pipeline::hasEmbeddings() {
    return model_->has_embeddings_;
}

void Pipeline::waitComplete() {
    timespec sleep_time{};
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = MILLISECOND; // check every 1 millisecond
    while (!isDone()) {
        nanosleep(&sleep_time, NULL);
    }
}

void PipelineCPU::addWorkersToPool(int pool_id, int worker_type, int num_workers) {
    bool *paused;
    ThreadStatus *status;

    for (int i = 0; i < num_workers; i++) {
        paused = new bool(true);
        status = new ThreadStatus(ThreadStatus::Paused);
        pool_paused_[pool_id]->emplace_back(paused);
        pool_status_[pool_id]->emplace_back(status);
        pool_[pool_id]->emplace_back(initThreadOfType(worker_type, paused, status, i));
    }
}

void PipelineCPU::initialize() {
    if (train_) {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, CPU_COMPUTE_ID, pipeline_options_->compute_threads);
        addWorkersToPool(2, CPU_ACCUMULATE_ID, pipeline_options_->gradient_update_threads);
        addWorkersToPool(3, UPDATE_EMBEDDINGS_ID, pipeline_options_->gradient_update_threads);
    } else {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, CPU_COMPUTE_ID, pipeline_options_->compute_threads);
    }
}

void PipelineCPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
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
        unaccumulated_batches_->expecting_data_ = expecting_data;
        update_batches_->expecting_data_ = expecting_data;
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
    }
}

PipelineGPU::PipelineGPU(DataLoader *dataloader,
                         std::shared_ptr<Model> model,
                         bool train,
                         ProgressReporter *reporter,
                         shared_ptr<PipelineConfig> pipeline_config) {
    dataloader_ = dataloader;
    model_ = model;
    reporter_ = reporter;
    train_ = train;
    edges_processed_ = 0;
    pipeline_options_ = pipeline_config;

    if (train_) {
        loaded_batches_ = new Queue<vector<Batch *>>(pipeline_options_->batch_host_queue_size);
        device_loaded_batches_ = new Queue<vector<Batch *>>(pipeline_options_->batch_device_queue_size);
        if (model_->has_embeddings_) {
            device_update_batches_ = new Queue <vector<Batch *>>(pipeline_options_->gradients_device_queue_size);
            update_batches_ = new Queue <vector<Batch *>>(pipeline_options_->gradients_host_queue_size);
        }
    }  else {
        loaded_batches_ = new Queue<vector<Batch *>>(pipeline_options_->batch_host_queue_size);
        device_loaded_batches_ = new Queue<vector<Batch *>>(pipeline_options_->batch_device_queue_size);
    }

    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<thread>;
    }

    staleness_bound_ = pipeline_options_->staleness_bound;
    batches_in_flight_ = 0;
    admitted_batches_ = 0;

    PipelineGPU::initialize();
}


PipelineGPU::~PipelineGPU() {
    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        delete pool_paused_[i];
        delete pool_status_[i];
        delete pool_[i];
    }

    if (train_) {
        delete loaded_batches_;
        delete device_loaded_batches_;
        if (model_->has_embeddings_) {
            delete device_update_batches_;
            delete update_batches_;
        }
    }  else {
        delete loaded_batches_;
        delete device_loaded_batches_;
    }
}

void PipelineGPU::addWorkersToPool(int pool_id, int worker_type, int num_workers) {
    bool *paused;
    ThreadStatus *status;

    for (int i = 0; i < num_workers; i++) {
        paused = new bool(true);
        status = new ThreadStatus(ThreadStatus::Paused);
        pool_paused_[pool_id]->emplace_back(paused);
        pool_status_[pool_id]->emplace_back(status);
        pool_[pool_id]->emplace_back(initThreadOfType(worker_type, paused, status, i));
    }
}

void PipelineGPU::initialize() {
    if (train_) {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, EMBEDDINGS_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
        addWorkersToPool(2, GPU_COMPUTE_ID, 1); // Only one thread manages GPU
        if (model_->has_embeddings_) {
            addWorkersToPool(3, UPDATE_TRANSFER_ID, pipeline_options_->gradient_transfer_threads);
            addWorkersToPool(4, UPDATE_EMBEDDINGS_ID, pipeline_options_->gradient_update_threads);
        }
    } else {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, EMBEDDINGS_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
        addWorkersToPool(2, GPU_COMPUTE_ID, 1);
    }
}

void PipelineGPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
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
        device_loaded_batches_->flush();
        if (model_->has_embeddings_) {
            device_update_batches_->flush();
            update_batches_->flush();
        }
    } else {
        loaded_batches_->flush();
        device_loaded_batches_->flush();
    }
}

void PipelineGPU::setQueueExpectingData(bool expecting_data) {
    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        device_loaded_batches_->expecting_data_ = expecting_data;
        if (model_->has_embeddings_) {
            device_update_batches_->expecting_data_ = expecting_data;
            update_batches_->expecting_data_ = expecting_data;
        }
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
        device_loaded_batches_->expecting_data_ = expecting_data;
    }
}
