//
// Created by Jason Mohoney on 2/29/20.
//

#include <pipeline.h>

using std::get;
using std::make_pair;
using std::forward_as_tuple;
using std::tie;

string getThreadStatusName(ThreadStatus status) {
    switch (status) {
        case ThreadStatus::Running:return "Running";
        case ThreadStatus::WaitPush:return "WaitPush";
        case ThreadStatus::WaitPop:return "WaitPop";
        case ThreadStatus::Paused:return "Paused";
        case ThreadStatus::Done:return "Done";
        default:return "Null";
    }
}

template<class T>
Queue<T>::Queue(uint max_size) {
    queue_ = std::deque<T>();
    max_size_ = max_size;
    mutex_ = new std::mutex();
    cv_ = new std::condition_variable();
    expecting_data_ = true;
}

Worker::Worker(Pipeline *pipeline, bool *paused, ThreadStatus *status) {
    pipeline_ = pipeline;
    sleep_time_.tv_sec = 0;
    sleep_time_.tv_nsec = WAIT_TIME;
    paused_ = paused;
    status_ = status;
}

void LoadEmbeddingsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            std::unique_lock lock(*pipeline_->max_batches_lock_);
            if ((pipeline_->batches_in_flight_ < pipeline_->max_batches_in_flight_) && pipeline_->admitted_batches_ < pipeline_->data_set_->getNumBatches()) {
                pipeline_->admitted_batches_++;
                pipeline_->batches_in_flight_++;
                lock.unlock();

                *status_ = ThreadStatus::Running;
                Batch *batch = pipeline_->data_set_->getBatch();
                if (batch == nullptr) {
                    break;
                }
                batch->timer_.start();
                SPDLOG_TRACE("Admitting Batch: {}", batch->batch_id_);

                Queue<Batch *> *push_queue = ((PipelineCPU *) pipeline_)->loaded_batches_;
                if (marius_options.general.device == torch::kCUDA) {
                    push_queue = ((PipelineGPU *) pipeline_)->loaded_batches_;
                }
                *status_ = ThreadStatus::WaitPush;
                push_queue->blocking_push(batch);
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

void EmbeddingsToDeviceWorker::run() {


    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;
            Queue<Batch *> *pop_queue = ((PipelineGPU *) pipeline_)->loaded_batches_;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }
            *status_ = ThreadStatus::Running;

            // transfer data to device
            // chose device with fewest batches in queue:
            int num_gpus = marius_options.general.gpu_ids.size();
            int device_id = 0;
            int min_size = ((PipelineGPU *) pipeline_)->device_loaded_batches_[0]->size();
            for (int i = 1; i < num_gpus; i++) {
                if (((PipelineGPU *) pipeline_)->device_loaded_batches_[i]->size() < min_size) {
                    min_size = ((PipelineGPU *) pipeline_)->device_loaded_batches_[i]->size();
                    device_id = i;
                }
            }

            at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, device_id);
            at::cuda::setCurrentCUDAStream(myStream);

            batch->embeddingsToDevice(device_id);
            pipeline_->data_set_->loadGPUParameters(batch);
            batch->prepareBatch();

            Queue<Batch *> *push_queue = ((PipelineGPU *) pipeline_)->device_loaded_batches_[device_id];
            *status_ = ThreadStatus::WaitPush;
            push_queue->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }

}

void PrepareBatchWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            Queue<Batch *> *pop_queue = ((PipelineCPU *) pipeline_)->loaded_batches_; // retrieves loaded batches pointer
            *status_ = ThreadStatus::WaitPop;
            auto tup = pop_queue->blocking_pop(); // get batch
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }
            *status_ = ThreadStatus::Running;

            pipeline_->data_set_->loadGPUParameters(batch);
            batch->prepareBatch();
            pipeline_->timestamp_lock_.lock();
            pipeline_->oldest_timestamp_ = batch->load_timestamp_;
            pipeline_->timestamp_lock_.unlock();

            Queue<Batch *> *push_queue = ((PipelineCPU *) pipeline_)->prepped_batches_;
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
            Queue<Batch *> *pop_queue = ((PipelineCPU *) pipeline_)->prepped_batches_;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            if (pipeline_->isTrain()) {
                pipeline_->model_->train(batch);
                batch->compute_timestamp_ = global_timestamp_allocator.getTimestamp();
                batch->status_ = BatchStatus::ComputedGradients;
                Queue<Batch *> *push_queue = ((PipelineCPU *) pipeline_)->unaccumulated_batches_;
                *status_ = ThreadStatus::WaitPush;
                push_queue->blocking_push(batch);
            } else {
                pipeline_->model_->evaluate(batch);
                pipeline_->data_set_->batches_processed_++;
                pipeline_->batches_in_flight_--;
                pipeline_->max_batches_cv_->notify_one();
                batch->clear();
            }
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

ComputeWorkerGPU::ComputeWorkerGPU(Pipeline *pipeline, int device_id, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {
    device_id_ = device_id;
}

void ComputeWorkerGPU::run() {

    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(true, device_id_);
    at::cuda::setCurrentCUDAStream(myStream);

    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            *status_ = ThreadStatus::WaitPop;

            Timer pop_time = Timer(false);
            Timer train_time_host = Timer(false);
            Timer train_time_device = Timer(true);
            Timer push_time = Timer(false);


            pop_time.start();
            Queue<Batch *> *pop_queue = ((PipelineGPU *) pipeline_)->device_loaded_batches_[device_id_];
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }
            pop_time.stop();
            SPDLOG_INFO("Pop Time: {}", pop_time.getDuration());


            train_time_host.start();
            train_time_device.start();
            *status_ = ThreadStatus::Running;
            if (pipeline_->isTrain()) {
                pipeline_->model_->train(batch);
                train_time_host.stop();
                train_time_device.stop();
                SPDLOG_INFO("Train Time Host: {}", train_time_host.getDuration());
                SPDLOG_INFO("Train Time Device: {}", train_time_device.getDuration());

                push_time.start();
                Queue<Batch *> *push_queue = ((PipelineGPU *) pipeline_)->device_update_batches_[device_id_];
                *status_ = ThreadStatus::WaitPush;
                push_queue->blocking_push(batch);
                push_time.stop();
                SPDLOG_INFO("Push Time: {}", push_time.getDuration());
            } else {
                pipeline_->model_->evaluate(batch);
                pipeline_->data_set_->batches_processed_++;
                pipeline_->batches_in_flight_--;
                pipeline_->max_batches_cv_->notify_one();
                batch->clear();
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
            batch->accumulateGradients();

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

            // transfer data to device
            // chose device with most batches in queue:
            int num_gpus = marius_options.general.gpu_ids.size();
            int device_id = 0;
            int max_size = ((PipelineGPU *) pipeline_)->device_update_batches_[0]->size();
            for (int i = 1; i < num_gpus; i++) {
                if (((PipelineGPU *) pipeline_)->device_update_batches_[i]->size() > max_size) {
                    max_size = ((PipelineGPU *) pipeline_)->device_update_batches_[i]->size();
                    device_id = i;
                }
            }

            at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool(false, device_id);
            at::cuda::setCurrentCUDAStream(myStream);

            Queue<Batch *> *pop_queue = ((PipelineGPU *) pipeline_)->device_update_batches_[device_id];
            *status_ = ThreadStatus::WaitPop;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }
            batch->accumulateGradients();
            pipeline_->data_set_->updateEmbeddingsForBatch(batch, true);
            batch->compute_timestamp_ = global_timestamp_allocator.getTimestamp();
            batch->status_ = BatchStatus::ComputedGradients;

            *status_ = ThreadStatus::Running;
            batch->embeddingsToHost();

            Queue<Batch *> *push_queue = ((PipelineGPU *) pipeline_)->update_batches_;
            *status_ = ThreadStatus::WaitPush;
            push_queue->blocking_push(batch);
        }
        *status_ = ThreadStatus::Paused;
        nanosleep(&sleep_time_, NULL);
    }
}

void UpdateEmbeddingsWorker::run() {
    while (*status_ != ThreadStatus::Done) {
        while (!*paused_) {
            Queue<Batch *> *pop_queue = ((PipelineCPU *) pipeline_)->update_batches_;
            if (marius_options.general.device == torch::kCUDA) {
                pop_queue = ((PipelineGPU *) pipeline_)->update_batches_;
            }
            *status_ = ThreadStatus::WaitPop;
            auto tup = pop_queue->blocking_pop();
            bool popped = get<0>(tup);
            Batch *batch = get<1>(tup);
            if (!popped) {
                break;
            }

            *status_ = ThreadStatus::Running;
            pipeline_->data_set_->updateEmbeddingsForBatch(batch, false);
            pipeline_->data_set_->updateTimestamp();
            pipeline_->data_set_->batches_processed_++;
            pipeline_->batches_in_flight_--;
            pipeline_->max_batches_cv_->notify_one();
            pipeline_->edges_processed_ += batch->batch_size_;
            batch->clear();
            SPDLOG_TRACE("Completed: {}", batch->batch_id_);

            int64_t num_batches_per_log = pipeline_->data_set_->getNumBatches() / marius_options.reporting.logs_per_epoch;

            if ((batch->batch_id_ + 1) % num_batches_per_log == 0) {
                double completion = ((double) batch->batch_id_ + 1) / pipeline_->data_set_->getNumBatches();
                SPDLOG_INFO("Total Edges Processed: {}, Percent Complete: {:.3f}", pipeline_->edges_processed_, completion);
                process_mem_usage();
            }
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

thread Pipeline::initThreadOfType(int worker_type, bool *paused, ThreadStatus *status, int device_id) {
    thread t;

    if (worker_type == EMBEDDINGS_LOADER_ID) {
        auto load_embeddings_func = [](LoadEmbeddingsWorker w) { w.run(); };
        t = thread(load_embeddings_func, LoadEmbeddingsWorker(this, paused, status));
    } else if (worker_type == EMBEDDINGS_TRANSFER_ID) {
        auto embeddings_to_device_func = [](EmbeddingsToDeviceWorker w) { w.run(); };
        t = thread(embeddings_to_device_func, EmbeddingsToDeviceWorker(this, paused, status));
    } else if (worker_type == CPU_BATCH_PREP_ID) {
        auto compute_func = [](PrepareBatchWorker w) { w.run(); };
        t = thread(compute_func, PrepareBatchWorker(this, paused, status));
    } else if (worker_type == CPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerCPU w) { w.run(); };
        t = thread(compute_func, ComputeWorkerCPU(this, paused, status));
    } else if (worker_type == GPU_COMPUTE_ID) {
        auto compute_func = [](ComputeWorkerGPU w) { w.run(); };
        t = thread(compute_func, ComputeWorkerGPU(this, device_id, paused, status));
    } else if (worker_type == CPU_ACCUMULATE_ID) {
        auto compute_func = [](AccumulateGradientsWorker w) { w.run(); };
        t = thread(compute_func, AccumulateGradientsWorker(this, paused, status));
    } else if (worker_type == UPDATE_TRANSFER_ID) {
        auto gradients_to_host_func = [](GradientsToHostWorker w) { w.run(); };
        t = thread(gradients_to_host_func, GradientsToHostWorker(this, device_id, paused, status));
    } else if (worker_type == UPDATE_EMBEDDINGS_ID) {
        auto update_embeddings_func = [](UpdateEmbeddingsWorker w) { w.run(); };
        t = thread(update_embeddings_func, UpdateEmbeddingsWorker(this, paused, status));
    }
    return t;
}

void Pipeline::reportMRR() {
    ranks_ = data_set_->accumulateRanks();
    double avg_ranks = ranks_.mean().item<double>();
    double mrr = ranks_.reciprocal().mean().item<double>();
    double auc = data_set_->accumulateAuc();
    double ranks1 = (double) ranks_.le(1).nonzero().size(0) / ranks_.size(0);
    double ranks5 = (double) ranks_.le(5).nonzero().size(0) / ranks_.size(0);
    double ranks10 = (double) ranks_.le(10).nonzero().size(0) / ranks_.size(0);
    double ranks20 = (double) ranks_.le(20).nonzero().size(0) / ranks_.size(0);
    double ranks50 = (double) ranks_.le(50).nonzero().size(0) / ranks_.size(0);
    double ranks100 = (double) ranks_.le(100).nonzero().size(0) / ranks_.size(0);

    SPDLOG_INFO("Num Eval Edges: {}", data_set_->getNumEdges());
    SPDLOG_INFO("Num Eval Batches: {}", data_set_->batches_processed_);
    SPDLOG_INFO("Auc: {:.3f}, Avg Ranks: {:.3f}, MRR: {:.3f}, Hits@1: {:.3f}, Hits@5: {:.3f}, Hits@10: {:.3f}, Hits@20: {:.3f}, Hits@50: {:.3f}, Hits@100: {:.3f}", auc, avg_ranks, mrr, ranks1, ranks5, ranks10,
                ranks20, ranks50, ranks100);

    auto uniques = torch::_unique2(ranks_, true, true, true);
    auto vals = get<0>(uniques);
    auto counts = get<2>(uniques);

    torch::Tensor hist_tens = counts;
    vector<int64_t> hist_vec{hist_tens.data_ptr<int64_t>(), hist_tens.data_ptr<int64_t>() + hist_tens.size(0)};
    std::stringstream result;
    std::copy(hist_vec.begin(), hist_vec.end(), std::ostream_iterator<int64_t>(result, " "));
    SPDLOG_DEBUG("Histogram: {} ", result.str());
}

PipelineCPU::PipelineCPU(DataSet *data_set, Model *model, bool train, struct timespec report_time) {
    data_set_ = data_set;
    model_ = model;
    train_ = train;
    edges_processed_ = 0;

    if (train_) {
        loaded_batches_ = new Queue<Batch *>(marius_options.training_pipeline.embeddings_host_queue_size);
        prepped_batches_ = new Queue<Batch *>(marius_options.training_pipeline.embeddings_host_queue_size);
        unaccumulated_batches_ = new Queue<Batch *>(marius_options.training_pipeline.embeddings_host_queue_size);
        update_batches_ = new Queue<Batch *>(marius_options.training_pipeline.num_embedding_update_threads);
    } else {
        loaded_batches_ = new Queue<Batch *>(marius_options.evaluation_pipeline.embeddings_host_queue_size);
        prepped_batches_ = new Queue<Batch *>(marius_options.evaluation_pipeline.embeddings_host_queue_size);
    }

    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<thread>;
        trace_[i] = new vector<ThreadStatus>;
    }

    max_batches_in_flight_ = marius_options.training_pipeline.max_batches_in_flight;
    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();
    batches_in_flight_ = 0;
    report_time_ = report_time;
    admitted_batches_ = 0;

    if (train_) {
        logger_ = spdlog::get("TrainPipeline");
    } else {
        logger_ = spdlog::get("EvalPipeline");
    }
}

PipelineCPU::~PipelineCPU() {
    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        delete pool_paused_[i];
        delete pool_status_[i];
        delete pool_[i];
        delete trace_[i];
    }

    if (train_) {
        delete loaded_batches_;
        delete prepped_batches_;
        delete unaccumulated_batches_;
        delete update_batches_;
    } else {
        delete loaded_batches_;
        delete prepped_batches_;
    }
}

bool Pipeline::isDone() {
    return (batches_in_flight_ <= 0) && admitted_batches_ == data_set_->getNumBatches();
}

bool Pipeline::isTrain() {
    return train_;
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
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, marius_options.training_pipeline.num_embedding_loader_threads);
        addWorkersToPool(1, CPU_BATCH_PREP_ID, marius_options.training_pipeline.num_compute_threads);
        addWorkersToPool(2, CPU_COMPUTE_ID, marius_options.training_pipeline.num_compute_threads);
        addWorkersToPool(3, CPU_ACCUMULATE_ID, marius_options.training_pipeline.num_compute_threads);
        addWorkersToPool(4, UPDATE_EMBEDDINGS_ID, marius_options.training_pipeline.num_embedding_update_threads);
    } else {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, marius_options.evaluation_pipeline.num_embedding_loader_threads);
        addWorkersToPool(1, CPU_BATCH_PREP_ID, marius_options.evaluation_pipeline.num_evaluate_threads);
        addWorkersToPool(2, CPU_COMPUTE_ID, marius_options.evaluation_pipeline.num_evaluate_threads);
    }
    auto monitor_func = [](PipelineMonitor w) { w.run(); };
    monitor_ = thread(monitor_func, PipelineMonitor(this, report_time_));
}

void PipelineCPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    setQueueExpectingData(true);
    for (uint i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        trace_[i]->clear();
        for (uint j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = false;
        }
    }
}

void PipelineCPU::stopAndFlush() {

    waitComplete();
    setQueueExpectingData(false);

    string worker_names[] = {"Batch Loading Worker", "Prepare Batch Worker", "CPU Compute Worker", "Accumulate Batch Worker", "Update Batch Worker"};
    for (uint i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        string val = worker_names[i];

        int64_t total_size = trace_[i]->size();
        int64_t num_paused = 0;
        int64_t num_pull = 0;
        int64_t num_push = 0;
        int64_t num_running = 0;

        for (int64_t j = 0; j < total_size; j++) {
            switch (trace_[i]->at(j)) {
                case ThreadStatus::Running:num_running++;
                    break;
                case ThreadStatus::WaitPush:num_push++;
                    break;
                case ThreadStatus::WaitPop:num_pull++;
                    break;
                case ThreadStatus::Paused:num_paused++;
                    break;
                case ThreadStatus::Done:
                default:break;
            }
        }
        SPDLOG_DEBUG("{} Paused: {}. WaitPop: {}. WaitPush: {}. Running: {}", val, (float) num_paused / total_size, (float) num_pull / total_size, (float) num_push / total_size, (float) num_running / total_size);
        for (uint j = 0; j < pool_paused_[i]->size(); j++) {
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
        prepped_batches_->flush();
        unaccumulated_batches_->flush();
        update_batches_->flush();
    } else {
        loaded_batches_->flush();
        prepped_batches_->flush();
    }
}

void PipelineCPU::setQueueExpectingData(bool expecting_data) {
    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        prepped_batches_->expecting_data_ = expecting_data;
        unaccumulated_batches_->expecting_data_ = expecting_data;
        update_batches_->expecting_data_ = expecting_data;
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
        prepped_batches_->expecting_data_ = expecting_data;
    }
}

void PipelineCPU::reportQueueStatus() {
    logger_->trace("############################");
    logger_->trace("Batches in Flight: {}", batches_in_flight_);
    logger_->trace("Batch Queue: {}/{}", loaded_batches_->size(), loaded_batches_->getMaxSize());
    logger_->trace("Prepped Batch Queue: {}/{}", prepped_batches_->size(), prepped_batches_->getMaxSize());

    if (train_) {
        logger_->trace("Unaccumulated Batch Queue: {}/{}", unaccumulated_batches_->size(), unaccumulated_batches_->getMaxSize());
        logger_->trace("Update Batch Queue: {}/{}", update_batches_->size(), update_batches_->getMaxSize());
    }
}

void PipelineCPU::reportThreadStatus() {
    string worker_names[] = {"Batch Loading Worker", "Prepare Batch Worker", "CPU Compute Worker", "Accumulate Batch Worker", "Update Batch Worker"};
    for (int i = 0; i < CPU_NUM_WORKER_TYPES; i++) {
        int tid = i;
        string val = worker_names[tid];
        string status = "[";

        for (uint j = 0; j < pool_status_[tid]->size(); j++) {
            ThreadStatus t_status = *pool_status_[tid]->at(j);
            status += getThreadStatusName(t_status);
            trace_[tid]->emplace_back(t_status);
            if (j + 1 < pool_status_[tid]->size()) {
                status += ", ";
            }
        }
        status += "]";
        logger_->trace("{} Status: {}", val, status);
    }
}

PipelineGPU::PipelineGPU(DataSet *data_set, Model *model, bool train, struct timespec report_time) {
    data_set_ = data_set;
    model_ = model;
    train_ = train;
    edges_processed_ = 0;
    int num_gpus = marius_options.general.gpu_ids.size();

    if (train_) {
        loaded_batches_ = new Queue<Batch *>(marius_options.training_pipeline.embeddings_host_queue_size);
        device_loaded_batches_ = std::vector<Queue<Batch *> *>(num_gpus);
        device_update_batches_ = std::vector<Queue<Batch *> *>(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i] = new Queue<Batch *>(marius_options.training_pipeline.embeddings_device_queue_size);
            device_update_batches_[i] = new Queue<Batch *>(marius_options.training_pipeline.gradients_device_queue_size);
        }
        update_batches_ = new Queue<Batch *>(marius_options.training_pipeline.gradients_host_queue_size);
    }  else {
        loaded_batches_ = new Queue<Batch *>(marius_options.evaluation_pipeline.embeddings_host_queue_size);
        device_loaded_batches_ = std::vector<Queue<Batch *> *>(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i] = new Queue<Batch *>(marius_options.evaluation_pipeline.embeddings_device_queue_size);
        }
    }

    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        pool_paused_[i] = new vector<bool *>;
        pool_status_[i] = new vector<ThreadStatus *>;
        pool_[i] = new vector<thread>;
        trace_[i] = new vector<ThreadStatus>;
    }

    max_batches_in_flight_ = marius_options.training_pipeline.max_batches_in_flight;
    batches_in_flight_ = 0;
    report_time_ = report_time;
    admitted_batches_ = 0;

    if (train_) {
        logger_ = spdlog::get("TrainPipeline");
    } else {
        logger_ = spdlog::get("EvalPipeline");
    }

}


PipelineGPU::~PipelineGPU() {
    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        delete pool_paused_[i];
        delete pool_status_[i];
        delete pool_[i];
        delete trace_[i];
    }

    int num_gpus = marius_options.general.gpu_ids.size();

    if (train_) {
        delete loaded_batches_;
        for (int i = 0; i < num_gpus; i++) {
            delete device_loaded_batches_[i];
            delete device_update_batches_[i];
        }
        delete update_batches_;
    }  else {
        delete loaded_batches_;
        for (int i = 0; i < num_gpus; i++) {
            delete device_loaded_batches_[i];
        }
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
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, marius_options.training_pipeline.num_embedding_loader_threads);
        addWorkersToPool(1, EMBEDDINGS_TRANSFER_ID, marius_options.training_pipeline.num_embedding_transfer_threads);
        addWorkersToPool(2, GPU_COMPUTE_ID, marius_options.training_pipeline.num_compute_threads);
        addWorkersToPool(3, UPDATE_TRANSFER_ID, marius_options.training_pipeline.num_gradient_transfer_threads);
        addWorkersToPool(4, UPDATE_EMBEDDINGS_ID, marius_options.training_pipeline.num_embedding_update_threads);
    } else {
        addWorkersToPool(0, EMBEDDINGS_LOADER_ID, marius_options.evaluation_pipeline.num_embedding_loader_threads);
        addWorkersToPool(1, EMBEDDINGS_TRANSFER_ID, marius_options.evaluation_pipeline.num_embedding_transfer_threads);
        addWorkersToPool(2, GPU_COMPUTE_ID, marius_options.evaluation_pipeline.num_evaluate_threads);
    }

    auto monitor_func = [](PipelineMonitor w) { w.run(); };
    monitor_ = thread(monitor_func, PipelineMonitor(this, report_time_));
}

void PipelineGPU::start() {
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    setQueueExpectingData(true);
    for (uint i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        trace_[i]->clear();
        for (uint j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = false;
        }
    }
}

void PipelineGPU::stopAndFlush() {

    waitComplete();
    setQueueExpectingData(false);

    string worker_names[] = {"Batch Loading Worker", "Batch H2D Transfer Worker", "GPU Compute Worker", "Batch D2H Transfer Worker", "Update Batch Worker"};
    for (uint i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        string val = worker_names[i];

        int64_t total_size = trace_[i]->size();
        int64_t num_paused = 0;
        int64_t num_pull = 0;
        int64_t num_push = 0;
        int64_t num_running = 0;

        for (int64_t j = 0; j < total_size; j++) {
            switch (trace_[i]->at(j)) {
                case ThreadStatus::Running:num_running++;
                    break;
                case ThreadStatus::WaitPush:num_push++;
                    break;
                case ThreadStatus::WaitPop:num_pull++;
                    break;
                case ThreadStatus::Paused:num_paused++;
                    break;
                case ThreadStatus::Done:
                default:break;
            }
        }
        SPDLOG_DEBUG("{} Paused: {}. WaitPop: {}. WaitPush: {}. Running: {}", val, (float) num_paused / total_size, (float) num_pull / total_size, (float) num_push / total_size, (float) num_running / total_size);
        for (uint j = 0; j < pool_paused_[i]->size(); j++) {
            *pool_paused_[i]->at(j) = true;
        }
    }
    max_batches_cv_->notify_all();

    SPDLOG_INFO("Pipeline flush complete");
    edges_processed_ = 0;
}

void PipelineGPU::flushQueues() {
    int num_gpus = marius_options.general.gpu_ids.size();

    if (train_) {
        loaded_batches_->flush();
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i]->flush();
            device_update_batches_[i]->flush();
        }
        update_batches_->flush();
    } else {
        loaded_batches_->flush();
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i]->flush();
        }
    }
}

void PipelineGPU::setQueueExpectingData(bool expecting_data) {
    int num_gpus = marius_options.general.gpu_ids.size();

    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i]->expecting_data_ = expecting_data;
            device_update_batches_[i]->expecting_data_ = expecting_data;
        }
        update_batches_->expecting_data_ = expecting_data;
    } else {
        loaded_batches_->expecting_data_ = expecting_data;
        for (int i = 0; i < num_gpus; i++) {
            device_loaded_batches_[i]->expecting_data_ = expecting_data;
        }
    }
}

void PipelineGPU::reportQueueStatus() {
    int num_gpus = marius_options.general.gpu_ids.size();

    logger_->trace("############################");
    logger_->trace("Batches in Flight: {}", batches_in_flight_);
    logger_->trace("Host Batch Queue: {}/{}", loaded_batches_->size(), loaded_batches_->getMaxSize());
    for (int i = 0; i < num_gpus; i++) {
        logger_->trace("Device {} Batch Queue: {}/{}", i, device_loaded_batches_[i]->size(), device_loaded_batches_[i]->getMaxSize());
        if (train_) {
            logger_->trace("Device {} Update Batch Queue: {}/{}", i, device_update_batches_[i]->size(), device_update_batches_[i]->getMaxSize());
        }
    }
    if (train_) {
    	logger_->trace("Host Update Batch Queue: {}/{}", update_batches_->size(), update_batches_->getMaxSize());
	}
}

void PipelineGPU::reportThreadStatus() {

    string worker_names[] = {"Batch Loading Worker", "Batch H2D Transfer Worker", "GPU Compute Worker", "Batch D2H Transfer Worker", "Update Batch Worker"};
    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        int tid = i;
        string val = worker_names[tid];
        string status = "[";

        for (uint j = 0; j < pool_status_[tid]->size(); j++) {
            ThreadStatus t_status = *pool_status_[tid]->at(j);
            status += getThreadStatusName(t_status);
            trace_[tid]->emplace_back(t_status);
            if (j + 1 < pool_status_[tid]->size()) {
                status += ", ";
            }
        }
        status += "]";
        logger_->trace("{} Status: {}", val, status);
    }
}

PipelineMonitor::PipelineMonitor(Pipeline *pipeline, struct timespec sleep_time) {
    pipeline_ = pipeline;
    sleep_time_ = sleep_time;
}

void PipelineMonitor::run() {
    while (true) {
        pipeline_->reportQueueStatus();
        pipeline_->reportThreadStatus();
        nanosleep(&sleep_time_, NULL);
    }
}
