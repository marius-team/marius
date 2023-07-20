//
// Created by Jason Mohoney on 1/21/22.
//

#include "pipeline/pipeline_gpu.h"

#include "pipeline/queue.h"
#include "reporting/logger.h"

void BatchToDeviceWorker::run() {
    unsigned int rand_seed = rand();

    int assign_id = 0;

    while (!done_) {
        while (!paused_) {
            auto tup = ((PipelineGPU *)pipeline_)->loaded_batches_->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            if (batch->sub_batches_.size() > 0) {
                #pragma omp parallel for
                for (int i = 0; i < batch->sub_batches_.size(); i++) {
                    batch->sub_batches_[i]->to(pipeline_->model_->device_models_[i]->device_, pipeline_->dataloader_->compute_streams_[i]);
//                    std::cout<<"to: "<<pipeline_->model_->device_models_[i]->device_<<"\n";
//                    ((PipelineGPU *)pipeline_)->device_loaded_batches_[i]->blocking_push(batch->sub_batches_[i]);
                }
            } else {
                batch->to(pipeline_->model_->device_models_[0]->device_, pipeline_->dataloader_->compute_streams_[0]);
            }

            ((PipelineGPU *)pipeline_)->device_loaded_batches_[0]->blocking_push(batch);

        }
        nanosleep(&sleep_time_, NULL);
    }
}

void ComputeWorkerGPU::run() {
//    at::cuda::CUDAStream compute_stream = at::cuda::getStreamFromPool(true, 0);
//    if (pipeline_->dataloader_->learning_task_ == LearningTask::NODE_CLASSIFICATION) {
//        pipeline_->dataloader_->compute_streams_[0] = &compute_stream;
//    }
//    //TODO: streams for LP need a bit more work



//    std::cout<<"start: "<<gpu_id_<<"\n";
    CudaStream compute_stream = getStreamFromPool(true, gpu_id_);
    pipeline_->dataloader_->compute_streams_[gpu_id_] = &compute_stream;



//    at::cuda::CUDAStreamGuard stream_guard(compute_stream);
//    std::cout<<compute_stream<<"\n";
//    at::cuda::setCurrentCUDAStream(compute_stream);

//    {
//        at::cuda::CUDAStream compute_stream = at::cuda::getStreamFromPool(true, 0);
//        pipeline_->dataloader_->compute_streams_[0] = &compute_stream;
//    }
//
//    for (int i = 0; i < 2; i++) {
//        at::cuda::CUDAStream compute_stream = at::cuda::getStreamFromPool(true, i);
//        pipeline_->dataloader_->compute_streams_[i] = &compute_stream;
//    }




    while (!done_) {
        while (!paused_) {
            auto tup = ((PipelineGPU *)pipeline_)->device_loaded_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

//            t.start();
            pipeline_->dataloader_->loadGPUParameters(batch); // TODO for sub_batches
//            t.stop();
//            std::cout<<"load: "<<t.getDuration()<<"\n";
//            std::cout<<"load\n";

            if (pipeline_->isTrain()) {
                bool will_sync = false;
//                if (pipeline_->model_->device_models_.size() > 1) {
//                    ((PipelineGPU *)pipeline_)->gpu_sync_lock_->lock();
//                    ((PipelineGPU *)pipeline_)->batches_since_last_sync_++;
//
//                    if (((PipelineGPU *)pipeline_)->batches_since_last_sync_ == ((PipelineGPU *)pipeline_)->gpu_sync_interval_) {
//                        will_sync = true;
//                    }
//
//                    // only release the lock if we don't need to synchronize the GPUs
//                    if (!will_sync) {
//                        ((PipelineGPU *)pipeline_)->gpu_sync_lock_->unlock();
//                    }
//                }

//                if (pipeline_->dataloader_->compute_streams_[0] != nullptr) {
//                    at::cuda::CUDAStreamGuard stream_guard(compute_stream);
//                    pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch, ((PipelineGPU *) pipeline_)->pipeline_options_->gpu_model_average);
//                } else {
//                    pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch, ((PipelineGPU *) pipeline_)->pipeline_options_->gpu_model_average);
//                }




                if (batch->sub_batches_.size() > 0) {
                    int i = gpu_id_;
//                    std::cout<<gpu_id_<<"\n";
//                    std::cout<<"on: "<<batch->node_features_.device()<<"\n";
//                    at::cuda::CUDAStreamGuard stream_guard(compute_stream);
//                    at::cuda::CUDAStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[gpu_id_]));
//                    pipeline_->model_->device_models_[1].get()->train_batch(batch->sub_batches_[1], true);


//                    t.start();
//                    at::cuda::CUDAMultiStreamGuard multi_guard({*(pipeline_->dataloader_->compute_streams_[0]),
//                                                                *(pipeline_->dataloader_->compute_streams_[1])});
//                    std::cout<<"guard\n";
//                    t.stop();
//                    std::cout<<"stream guard: "<<t.getDuration()<<"\n";
//                    pipeline_->model_->clear_grad_all();

//                    t.start();
                    #pragma omp parallel
                    {
                        #pragma omp for
                        for (int i = 0; i < batch->sub_batches_.size(); i++) {
                            CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                            pipeline_->model_->device_models_[i]->clear_grad();
                            pipeline_->model_->device_models_[i]->train_batch(batch->sub_batches_[i], false);

//                            pipeline_->model_->device_models_[i]->step();
//                            pipeline_->model_->device_models_[i]->train_batch(batch, true);
//                            std::cout<<"train_batch: "<<i<<"\n";
                        }


                        #pragma omp single
                        {
//                            at::cuda::setCurrentCUDAStream(*(pipeline_->dataloader_->compute_streams_[0]);
//                            at::cuda::setCurrentCUDAStream(*(pipeline_->dataloader_->compute_streams_[1]));
                            CudaMultiStreamGuard multi_guard({*(pipeline_->dataloader_->compute_streams_[0]),
                                                              *(pipeline_->dataloader_->compute_streams_[1])}); // TODO: general multi-gpu
//                            std::vector<at::cuda::CUDAStream *> streams = {pipeline_->dataloader_->compute_streams_[0],
//                                                                           pipeline_->dataloader_->compute_streams_[1]};
//                            pipeline_->model_->all_reduce(streams);

                            pipeline_->model_->all_reduce();
////                            std::cout<<"all reduce\n";
                        }

                        #pragma omp for
                        for (int i = 0; i < batch->sub_batches_.size(); i++) {
                            CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                            pipeline_->model_->device_models_[i]->step();
//                            std::cout<<"step: "<<i<<"\n";
                        }
                    }

//                    t.stop();
//                    std::cout<<"train_batch: "<<t.getDuration()<<"\n";

//                    t.start();
//                    pipeline_->model_->all_reduce();
//                    t.stop();
//                    std::cout<<"all_reduce: "<<t.getDuration()<<"\n";

//                    t.start();
//                    pipeline_->model_->step_all();
//                    t.stop();
//                    std::cout<<"step: "<<t.getDuration()<<"\n";


                } else {
                    CudaStreamGuard stream_guard(compute_stream);
                    pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch, true);
                }







//                if (will_sync) {
//                    // we already have the lock acquired, it is safe to sync?
//                    pipeline_->model_->all_reduce();
//
//                    ((PipelineGPU *)pipeline_)->batches_since_last_sync_ = 0;
//                    ((PipelineGPU *)pipeline_)->gpu_sync_lock_->unlock();
//                }

//                t.start();
                if (!pipeline_->has_embeddings()) {
                    batch->clear();
                    pipeline_->reporter_->addResult(batch->batch_size_);
                    pipeline_->batches_in_flight_--;
                    pipeline_->dataloader_->finishedBatch();
                    pipeline_->max_batches_cv_->notify_one();
                    pipeline_->edges_processed_ += batch->batch_size_;
                } else {
                    pipeline_->dataloader_->updateEmbeddings(batch, true); // TODO: if sub_batches
                    ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
                }
            } else {
                pipeline_->model_->device_models_[gpu_id_]->evaluate_batch(batch);

                pipeline_->batches_in_flight_--;
                pipeline_->max_batches_cv_->notify_one();
                pipeline_->dataloader_->finishedBatch();
                batch->clear();
            }
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void EncodeNodesWorkerGPU::run() {
    while (!done_) {
        while (!paused_) {
            auto tup = ((PipelineGPU *)pipeline_)->device_loaded_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            pipeline_->dataloader_->loadGPUParameters(batch);

            torch::Tensor encoded =
                pipeline_->model_->device_models_[gpu_id_].get()->encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
            batch->clear();
            batch->encoded_uniques_ = encoded.contiguous();

            ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void BatchToHostWorker::run() {
    while (!done_) {
        while (!paused_) {
            auto tup = ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_pop();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            if (batch->sub_batches_.size() > 0) {
                #pragma omp parallel for
                for (int i = 0; i < batch->sub_batches_.size(); i++) {
                    CudaStream transfer_stream = getStreamFromPool(false, i);
                    CudaStreamGuard stream_guard(transfer_stream);
                    batch->sub_batches_[i]->embeddingsToHost();
                }
            }
            else {
                CudaStream transfer_stream = getStreamFromPool(false, gpu_id_);
                CudaStreamGuard stream_guard(transfer_stream);
                batch->embeddingsToHost();
            }

            ((PipelineGPU *)pipeline_)->update_batches_->blocking_push(batch);
        }
        nanosleep(&sleep_time_, NULL);
    }
}

PipelineGPU::PipelineGPU(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, bool train, shared_ptr<ProgressReporter> reporter,
                         shared_ptr<PipelineConfig> pipeline_config, bool encode_only) {
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
    } else {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        device_loaded_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_device_queue_size));
    }

    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();

    staleness_bound_ = pipeline_options_->staleness_bound;
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    curr_pos_ = 0;

    PipelineGPU::initialize();
}

PipelineGPU::~PipelineGPU() {
    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_[i].size(); j++) {
            pool_[i][j]->stop();
        }
    }

    pool_->clear();

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
    for (int i = 0; i < num_workers; i++) {
        for (int j = 0; j < num_gpus; j++) {
            pool_[pool_id].emplace_back(initWorkerOfType(worker_type, j, i));
        }
    }
}

void PipelineGPU::initialize() {
    if (encode_only_) {
        addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
        addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
        addWorkersToPool(2, GPU_ENCODE_ID, 1, model_->device_models_.size());  // Only one std::thread manages GPU
        if (model_->has_embeddings()) {
            addWorkersToPool(3, D2H_TRANSFER_ID, pipeline_options_->gradient_transfer_threads, model_->device_models_.size());
            addWorkersToPool(4, NODE_WRITE_ID, pipeline_options_->gradient_update_threads);
        }
    } else {
        if (train_) {
            addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);
            addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
            addWorkersToPool(2, GPU_COMPUTE_ID, 1, model_->device_models_.size());  // Only one std::thread manages GPU
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
        for (int j = 0; j < pool_[i].size(); j++) {
            pool_[i][j]->start();
        }
    }
}

void PipelineGPU::pauseAndFlush() {
    waitComplete();
    setQueueExpectingData(false);

    for (int i = 0; i < GPU_NUM_WORKER_TYPES; i++) {
        for (int j = 0; j < pool_[i].size(); j++) {
            pool_[i][j]->pause();
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
            for (auto d : device_update_batches_) {
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
