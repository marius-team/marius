//
// Created by Jason Mohoney on 1/21/22.
//

#ifdef MARIUS_CUDA
    #include <torch/csrc/cuda/nccl.h>
#endif

#include "pipeline/pipeline_cpu.h"
#include "pipeline/pipeline_gpu.h"

#include "pipeline/queue.h"
#include "reporting/logger.h"

void batchToDevice(Pipeline* pipeline_, shared_ptr<Batch> batch) {
    if (batch->sub_batches_.size() > 0) {
        int chunk_size = batch->sub_batches_.size() / batch->num_sub_batches_;
        for (int j = 0; j < batch->num_sub_batches_; j++) {
            #pragma omp parallel for
            for (int i = 0; i < chunk_size; i++) {
                batch->sub_batches_[i+chunk_size*j]->to(pipeline_->model_->device_models_[i]->device_, pipeline_->dataloader_->compute_streams_[i]);
            }
        }
    } else {
        batch->to(pipeline_->model_->device_models_[0]->device_, pipeline_->dataloader_->compute_streams_[0]);
    }

    ((PipelineGPU *)pipeline_)->device_loaded_batches_[0]->blocking_push(batch);

//  std::cout<<"to: "<<pipeline_->model_->device_models_[i]->device_<<"\n";
//  ((PipelineGPU *)pipeline_)->device_loaded_batches_[i]->blocking_push(batch->sub_batches_[i]);
}

void updateEvalForBatch(Pipeline* pipeline_, shared_ptr<Batch> batch) {
    if (batch->neg_scores_.defined()) {
        std::dynamic_pointer_cast<LinkPredictionReporter>(pipeline_->model_->reporter_)->addResult(batch->pos_scores_, batch->neg_scores_);
    }
    if (batch->inv_neg_scores_.defined()) {
        std::dynamic_pointer_cast<LinkPredictionReporter>(pipeline_->model_->reporter_)->addResult(batch->inv_pos_scores_, batch->inv_neg_scores_);
    }
    if (batch->y_pred_.defined()) {
        std::dynamic_pointer_cast<NodeClassificationReporter>(pipeline_->model_->reporter_)->addResult(batch->node_labels_, batch->y_pred_);
    }

    pipeline_->batches_in_flight_--;
    pipeline_->max_batches_cv_->notify_one();
    pipeline_->dataloader_->finishedBatch();
    batch->clear();
}

void RemoteLoadWorker::run() {
    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            // NOTE: this "train" is probably not set correctly all the time
            shared_ptr<Batch> batch = std::make_shared<Batch>(pipeline_->dataloader_->train_);

            std::unique_lock lock(*pipeline_->fwd_tag_lock_);
            int parent_id = pipeline_->fwd_round_robin_parent_ % pipeline_->model_->parents_.size();
            int parent = pipeline_->model_->parents_[parent_id];
            int tag = pipeline_->fwd_parent_tags_[parent_id];
            pipeline_->fwd_parent_tags_[parent_id] = pipeline_->fwd_parent_tags_[parent_id] + 8;
            pipeline_->fwd_round_robin_parent_++;
            lock.unlock();

            if (parent == pipeline_->model_->pg_gloo_->pg->getRank()) { // parent is self
                // if the parent is self, then we don't need to receive batches. The batch loader workers will already
                // be putting batches on the loaded_batches_ queue
                continue;
            }

            if (pipeline_->model_->device_models_.size() > 1 and pipeline_->isTrain()) {
                std::vector <shared_ptr<Batch>> sub_batches;
                for (int i = 0; i < pipeline_->model_->device_models_.size() * pipeline_->dataloader_->num_sub_batches_; i++) {
                    shared_ptr<Batch> sub_batch = std::make_shared<Batch>(pipeline_->dataloader_->train_);
                    sub_batches.emplace_back(sub_batch);
                }
                batch->sub_batches_ = sub_batches;
            }

//            Timer t = new Timer(false);
//            t.start();
            batch->remoteReceive(pipeline_->model_->pg_gloo_->pg, parent, tag);
//            t.stop();
//            std::cout<<"remote load: "<<t.getDuration()<<"\n";
//            if (batch->sub_batches_.size() > 0) {
//                for (int i = 0; i < batch->sub_batches_.size(); i++) {
//                    batch->sub_batches_[i]->node_features_ = pipeline_->dataloader_->graph_storage_->getNodeFeatures(batch->sub_batches_[i]->unique_node_indices_);
//                }
//            } else {
//                batch->node_features_ = pipeline_->dataloader_->graph_storage_->getNodeFeatures(batch->unique_node_indices_);
//            }

            if (pipeline_->model_->device_.is_cuda()) {
                ((PipelineGPU *)pipeline_)->loaded_batches_->blocking_push(batch);
            } else {
                ((PipelineCPU *)pipeline_)->loaded_batches_->blocking_push(batch);
            }
//            t.stop();
//            std::cout<<"remote load: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void RemoteToDeviceWorker::run() {
    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->loaded_batches_->blocking_pop();
//            t.stop();
//            std::cout<<"remote to block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            std::unique_lock lock(*pipeline_->fwd_tag_lock_);
            int child_id = pipeline_->fwd_round_robin_child_ % pipeline_->model_->children_.size();
            int child = pipeline_->model_->children_[child_id];
            int tag = pipeline_->fwd_children_tags_[child_id];
            pipeline_->fwd_children_tags_[child_id] = pipeline_->fwd_children_tags_[child_id] + 8;
            pipeline_->fwd_round_robin_child_++;
            lock.unlock();

            if (child == pipeline_->model_->pg_gloo_->pg->getRank()) { // child is self
                // need to call regular to device here
                batchToDevice(pipeline_, batch); // TODO this would need to load cpu parameters given recent changes
                continue;
            }

            pipeline_->dataloader_->loadCPUParameters(batch, worker_id_, false);
            batch->creator_id_ = pipeline_->model_->pg_gloo_->pg->getRank();
            batch->remoteTo(pipeline_->model_->pg_gloo_->pg, child, tag);
//            t.stop();
//            std::cout<<"remote to: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void BatchSliceWorker::run() {
    unsigned int rand_seed = rand();

    int assign_id = 0;

    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->loaded_batches_->blocking_pop();
//            t.stop();
//            std::cout<<"batch to block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }


//            if (batch->sub_batches_.size() > 0) {
//                if (!batch->sub_batches_[0]->node_features_.defined()) {
//                    pipeline_->dataloader_->loadCPUParameters(batch);
//                }
//            } else {
//                if (!batch->node_features_.defined())
//                    pipeline_->dataloader_->loadCPUParameters(batch);
////                    batch->node_features_ = pipeline_->dataloader_->graph_storage_->getNodeFeatures(batch->unique_node_indices_);
////                    batch->node_labels_ = pipeline_->dataloader_->graph_storage_->getNodeLabels(
////                            batch->dense_graph_.node_ids_.narrow(0, batch->dense_graph_.hop_offsets_[-2].item<int64_t>(),
////                                                                 (batch->dense_graph_.node_ids_.size(0)-batch->dense_graph_.hop_offsets_[-2]).item<int64_t>())).flatten(0, 1);
//            }
//            pipeline_->dataloader_->loadCPUParameters(batch, worker_id_);

            ((PipelineGPU *)pipeline_)->loaded_sliced_batches_->blocking_push(batch);

//            t.stop();
//            std::cout<<"batch slice: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void BatchToDeviceWorker::run() {
    unsigned int rand_seed = rand();

    int assign_id = 0;

    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->loaded_sliced_batches_->blocking_pop();
//            t.stop();
//            std::cout<<"batch to block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }


//            if (batch->sub_batches_.size() > 0) {
//                if (!batch->sub_batches_[0]->node_features_.defined()) {
//                    pipeline_->dataloader_->loadCPUParameters(batch);
//                }
//            } else {
//                if (!batch->node_features_.defined())
//                    pipeline_->dataloader_->loadCPUParameters(batch);
////                    batch->node_features_ = pipeline_->dataloader_->graph_storage_->getNodeFeatures(batch->unique_node_indices_);
////                    batch->node_labels_ = pipeline_->dataloader_->graph_storage_->getNodeLabels(
////                            batch->dense_graph_.node_ids_.narrow(0, batch->dense_graph_.hop_offsets_[-2].item<int64_t>(),
////                                                                 (batch->dense_graph_.node_ids_.size(0)-batch->dense_graph_.hop_offsets_[-2]).item<int64_t>())).flatten(0, 1);
//            }
            pipeline_->dataloader_->loadCPUParameters(batch, worker_id_);
//            t.stop();
//            std::cout<<"batch load: "<<t.getDuration()<<"\n";

//            t.start();
            batchToDevice(pipeline_, batch);
//            t.stop();
//            std::cout<<"batch to: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void ComputeWorkerGPU::run() {
    CudaStream compute_stream = getStreamFromPool(true, gpu_id_);
    pipeline_->dataloader_->compute_streams_[gpu_id_] = &compute_stream;

    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->device_loaded_batches_[gpu_id_]->blocking_pop();
//            t.stop();
//            std::cout<<"compute block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            pipeline_->dataloader_->loadGPUParameters(batch); // TODO: this needs to be updated for multi-gpu (sub_batches)

            if (pipeline_->isTrain()) {
                // train batch
                if (batch->sub_batches_.size() > 0) {
//                    int i = gpu_id_;

                    std::vector<CudaStream> streams_for_multi_guard;
                    for (int i = 0; i < pipeline_->dataloader_->graph_storage_->num_gpus_; i++) {
                        streams_for_multi_guard.emplace_back(*(pipeline_->dataloader_->compute_streams_[i]));
                    }


                    int unique_size = 0;
                    int feat_dim = batch->sub_batches_[0]->node_features_.size(1);
                    int root_dim = batch->sub_batches_[0]->root_node_indices_.size(0);
//                    for (int i = 0; i < batch->sub_batches_.size(); i++) {
//                        unique_size += batch->sub_batches_[i]->node_features_.size(0);
//                    }

                    int chunk_size = batch->sub_batches_.size() / batch->num_sub_batches_;

                    unique_size = batch->sub_batches_[0]->node_features_.size(0) * chunk_size;
                    std::vector<torch::Tensor> inputs(chunk_size);
                    std::vector<torch::Tensor> unique_features_per_gpu(chunk_size);
                    std::vector<std::vector<torch::Tensor>> unique_gathered_features_per_gpu(chunk_size);
                    std::vector<torch::Tensor> broadcast_list(chunk_size);

//                    std::cout<<"start"<<"\n";
//                    std::cout<<unique_size<<"\n";
//                    std::cout<<feat_dim<<"\n";
//                    std::cout<<"NUM_SUB_BATCHES: "<<batch->num_sub_batches_<<"\n";
//                    std::cout<<batch->sub_batches_.size()<<" "<<batch->num_sub_batches_<<"\n";
//                    std::cout<<"chunk size"<<chunk_size<<"\n";


                    for (int j = 0; j < batch->num_sub_batches_; j++) {
                        #pragma omp parallel
                        {
                            #pragma omp for
                            for (int i = 0; i < chunk_size; i++) {
                                CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                                auto device_options = torch::TensorOptions().dtype(batch->sub_batches_[i+chunk_size*j]->node_features_.dtype()).device(batch->sub_batches_[i+chunk_size*j]->node_features_.device());

                                torch::Tensor unique_node_features = torch::zeros({unique_size, feat_dim}, device_options);
    ////                            std::cout<<unique_node_features.sizes()<<"\n";
    //
    //                            int count = 0;
    //                            for (int j = 0; j < batch->sub_batches_.size(); j++) {
    //                                unique_node_features.narrow(0, count, batch->sub_batches_[j]->node_features_.size(0)).copy_(batch->sub_batches_[j]->node_features_);
    //                                count += batch->sub_batches_[j]->node_features_.size(0);
    ////                                std::cout<<unique_node_features.sizes()<<"\n";
    //                                unique_features_per_gpu[i*batch->sub_batches_.size() + j] = torch::zeros({batch->sub_batches_[j]->node_features_.size(0), feat_dim}, device_options);
    //                            }

                                unique_features_per_gpu[i] = unique_node_features;
                                inputs[i] = batch->sub_batches_[i+chunk_size*j]->node_features_;

                                if (j == 0) {
                                    device_options = torch::TensorOptions().dtype(batch->sub_batches_[0]->root_node_indices_.dtype()).device(batch->sub_batches_[i]->node_features_.device());
                                    if (i > 0)
                                        broadcast_list[i] = torch::zeros({root_dim}, device_options);
                                    else
                                        broadcast_list[i] = batch->sub_batches_[i]->root_node_indices_;
                                }
                            }

                            #pragma omp single
                            {
                                CudaMultiStreamGuard multi_guard(streams_for_multi_guard);

                                #ifdef MARIUS_CUDA
                                    torch::cuda::nccl::all_gather(inputs, unique_features_per_gpu);//, streams);
                                    if (j == 0)
                                        torch::cuda::nccl::broadcast(broadcast_list);//, streams);
                                #endif

    //                            for (int j = 0; j < batch->sub_batches_.size(); j++) {
    //                                if (!device_models_[j]->named_parameters()[key].mutable_grad().defined()) {
    //                                    device_models_[j]->named_parameters()[key].mutable_grad() = torch::zeros_like(device_models_[j]->named_parameters()[key]);
    //                                }
    //                                // this line for averaging
    //                                device_models_[j]->named_parameters()[key].mutable_grad() /= (float_t) num_gpus;
    //
    //                                input_gradients[j] = device_models_[j]->named_parameters()[key].mutable_grad();
    //                            }
                            }

                            #pragma omp for
                            for (int i = 0; i < chunk_size; i++) {
                                CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                                unique_gathered_features_per_gpu[i].emplace_back(unique_features_per_gpu[i]);
                            }
                        }
                    }

                    if (batch->num_sub_batches_ > 1) {
                        #pragma omp for
                        for (int i = 0; i < chunk_size; i++) {
                            CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                            unique_features_per_gpu[i] = torch::cat({unique_gathered_features_per_gpu[i]}, 0);
                        }
                    }

                    // Train on each chunk of sub batches, one at a time
                    for (int j = 0; j < batch->num_sub_batches_; j++) {
                        #pragma omp parallel
                        {
                            #pragma omp for
                            for (int i = 0; i < chunk_size; i++) {
                                CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                                auto device_options = torch::TensorOptions().dtype(batch->sub_batches_[i+chunk_size*j]->node_features_.dtype()).device(batch->sub_batches_[i+chunk_size*j]->node_features_.device());

                                batch->sub_batches_[i+chunk_size*j]->unique_node_indices_ = torch::searchsorted(broadcast_list[i], batch->sub_batches_[i+chunk_size*j]->unique_node_indices_);

    //                            batch->sub_batches_[i]->node_features_ = torch::zeros({batch->sub_batches_[i]->unique_node_indices_.size(0), feat_dim}, device_options);
    //                            torch::index_select_out(batch->sub_batches_[i]->node_features_, unique_features_per_gpu[i], 0, batch->sub_batches_[i]->unique_node_indices_);
    //                            std::cout<<batch->sub_batches_[i]->node_features_.sizes()<<"\n";
                                batch->sub_batches_[i+chunk_size*j]->node_features_ = unique_features_per_gpu[i].index_select(0, batch->sub_batches_[i+chunk_size*j]->unique_node_indices_);

                                pipeline_->model_->device_models_[i]->clear_grad();
                                pipeline_->model_->device_models_[i]->train_batch(batch->sub_batches_[i+chunk_size*j], false);
                            }

                            #pragma omp single
                            {
                                CudaMultiStreamGuard multi_guard(streams_for_multi_guard);
                                pipeline_->model_->all_reduce();
                                pipeline_->model_->distGradSync();
                            }

                            #pragma omp for
                            for (int i = 0; i < chunk_size; i++) {
                                CudaStreamGuard stream_guard(*(pipeline_->dataloader_->compute_streams_[i]));
                                pipeline_->model_->device_models_[i]->step();
                            }

                            #pragma omp single
                            {
                                // TODO: should this be on this stream?
                                CudaMultiStreamGuard multi_guard(streams_for_multi_guard);
                                pipeline_->model_->distModelSync();
                            }
                        }
                    }

                    //TODO: here we should undo the uniques for the node embedding transfer back?

                } else {
                    CudaStreamGuard stream_guard(compute_stream);
                    pipeline_->model_->device_models_[gpu_id_].get()->train_batch(batch);
                }

                if (!pipeline_->has_embeddings()) {
                    // training: node classification
                    batch->clear();

                    if (pipeline_->compute_worker_needs_remote_) {
                        ((PipelineGPU *)pipeline_)->update_batches_->blocking_push(batch);
                    } else {
                        pipeline_->reporter_->addResult(batch->batch_size_, batch->getLoss(pipeline_->model_->model_config_->loss->options->loss_reduction));
                        pipeline_->batches_in_flight_--;
                        pipeline_->dataloader_->finishedBatch();
                        pipeline_->max_batches_cv_->notify_one();
                        pipeline_->edges_processed_ += batch->batch_size_;
                    }
                } else {
                    // training: link prediction
                    if (pipeline_->dataloader_->batch_worker_) {
                        pipeline_->dataloader_->updateEmbeddings(batch, true); // TODO: this needs to be updated for multi-gpu (sub_batches)
                    }
                    ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
                }
            } else {
                // evaluation
                if (pipeline_->compute_worker_needs_remote_) {
                    pipeline_->model_->device_models_[gpu_id_].get()->evaluate_batch(batch, true);
                    batch->clear(false);

                    ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_push(batch);
                } else {
                    pipeline_->model_->device_models_[gpu_id_].get()->evaluate_batch(batch);

                    pipeline_->batches_in_flight_--;
                    pipeline_->max_batches_cv_->notify_one();
                    pipeline_->dataloader_->finishedBatch();
                    batch->clear();
                }
            }
//            t.stop();
//            std::cout<<"compute: "<<t.getDuration()<<"\n";
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
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->device_update_batches_[gpu_id_]->blocking_pop();
//            t.stop();
//            std::cout<<"batch to host block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            if (pipeline_->isTrain()) {
                if (batch->sub_batches_.size() > 0) {
                    #pragma omp parallel for
                    for (int i = 0; i < batch->sub_batches_.size(); i++) {
                        CudaStream transfer_stream = getStreamFromPool(false, i);
                        CudaStreamGuard stream_guard(transfer_stream);
                        batch->sub_batches_[i]->embeddingsToHost();
                    }
                } else {
                    CudaStream transfer_stream = getStreamFromPool(false, gpu_id_);
                    CudaStreamGuard stream_guard(transfer_stream);
                    batch->embeddingsToHost();
                }
            } else {
                batch->evalToHost();
            }

            ((PipelineGPU *)pipeline_)->update_batches_->blocking_push(batch);
//            t.stop();
//            std::cout<<"batch to host: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void RemoteToHostWorker::run() {
    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            auto tup = ((PipelineGPU *)pipeline_)->update_batches_->blocking_pop();
//            t.stop();
//            std::cout<<"remote to host block: "<<t.getDuration()<<"\n";
//            t.start();
            bool popped = std::get<0>(tup);
            shared_ptr<Batch> batch = std::get<1>(tup);
            if (!popped) {
                break;
            }

            if (batch->creator_id_ == -1 or batch->creator_id_ == pipeline_->model_->pg_gloo_->pg->getRank()) { // parent is self
                if (pipeline_->isTrain()) {
                    // regular update batch for link prediction or notify to data loader that node classification batch finished
                    pipeline_->dataloader_->updateEmbeddings(batch, false);
                    batch->clear();

                    pipeline_->reporter_->addResult(batch->batch_size_, batch->getLoss(pipeline_->model_->model_config_->loss->options->loss_reduction));
                    pipeline_->batches_in_flight_--;
                    pipeline_->dataloader_->finishedBatch();
                    pipeline_->max_batches_cv_->notify_one();
                    pipeline_->edges_processed_ += batch->batch_size_;
                } else {
                    // eval
                    updateEvalForBatch(pipeline_, batch);
                }
                continue;
            }

            std::unique_lock lock(*pipeline_->bwd_tag_lock_);
            int parent_id = -1;
            for (int i = 0; i < pipeline_->model_->parents_.size(); i++) {
                if (pipeline_->model_->parents_[i] == batch->creator_id_)
                    parent_id = i;
            }
            int parent = batch->creator_id_;
            int tag = pipeline_->bwd_parent_tags_[parent_id];
            pipeline_->bwd_parent_tags_[parent_id] = pipeline_->bwd_parent_tags_[parent_id] + 8;
            lock.unlock();

            batch->remoteTo(pipeline_->model_->pg_gloo_->pg, parent, tag);
//            t.stop();
//            std::cout<<"remote to host: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

void RemoteListenForUpdatesWorker::run() {
    while (!done_) {
        while (!paused_) {
//            Timer t = new Timer(false);
//            t.start();
            // NOTE: this "train" is probably not set correctly all the time
            shared_ptr<Batch> batch = std::make_shared<Batch>(pipeline_->dataloader_->train_);

            std::unique_lock lock(*pipeline_->bwd_tag_lock_);
            int child_id = pipeline_->bwd_round_robin_child_ % pipeline_->model_->children_.size();
            int child = pipeline_->model_->children_[child_id];
            int tag = pipeline_->bwd_children_tags_[child_id];
            pipeline_->bwd_children_tags_[child_id] = pipeline_->bwd_children_tags_[child_id] + 8;
            pipeline_->bwd_round_robin_child_++;
            lock.unlock();

            if (child == pipeline_->model_->pg_gloo_->pg->getRank()) { // child is self
                // if the child is self, then we don't need to receive batch updates. The compute/device to host workers
                // will already be putting batches on the update_batches_ queue
                continue;
            }

            // NC batch finished notifications won't send batch sub_batches_ even if there were some,
            // they are cleared as they don't contain useful information. Eval also doesn't contain sub batches
            if (pipeline_->model_->device_models_.size() > 1 and pipeline_->has_embeddings() and pipeline_->isTrain()) {
                std::vector <shared_ptr<Batch>> sub_batches;
                for (int i = 0; i < pipeline_->model_->device_models_.size(); i++) {
                    shared_ptr<Batch> sub_batch = std::make_shared<Batch>(pipeline_->dataloader_->train_);
                    sub_batches.emplace_back(sub_batch);
                }
                batch->sub_batches_ = sub_batches;
            }

            batch->remoteReceive(pipeline_->model_->pg_gloo_->pg, child, tag);

            ((PipelineGPU *)pipeline_)->update_batches_->blocking_push(batch);
//            t.stop();
//            std::cout<<"remote listen: "<<t.getDuration()<<"\n";
        }
        nanosleep(&sleep_time_, NULL);
    }
}

PipelineGPU::PipelineGPU(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, bool train, shared_ptr<ProgressReporter> reporter,
                         shared_ptr<PipelineConfig> pipeline_config, bool encode_only,
                         bool batch_worker, bool compute_worker, bool batch_worker_needs_remote, bool compute_worker_needs_remote) {
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

    batch_worker_ = batch_worker;
    compute_worker_ = compute_worker;
    batch_worker_needs_remote_ = batch_worker_needs_remote;
    compute_worker_needs_remote_ = compute_worker_needs_remote;

    // Note that sometimes we don't actually need e.g., device_loaded_batches (for a batch worker but not compute worker)
    if (train_) {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        loaded_sliced_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_sliced_queue_size);
        for (int i = 0; i < model_->device_models_.size(); i++) {
            device_loaded_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_device_queue_size));
            if (model_->has_embeddings()) {
                device_update_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_device_queue_size));
            }
        }

        if (model_->has_embeddings() or (batch_worker_ and batch_worker_needs_remote_) or (compute_worker_ and compute_worker_needs_remote_)) {
            update_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_host_queue_size);
        }
    } else {
        loaded_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_host_queue_size);
        loaded_sliced_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_sliced_queue_size);
        for (int i = 0; i < model_->device_models_.size(); i++) {
            device_loaded_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->batch_device_queue_size));
            if (compute_worker_needs_remote_) {
                device_update_batches_.emplace_back(std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_device_queue_size));
            }
        }

        if (compute_worker_needs_remote_ or batch_worker_needs_remote_) {
            update_batches_ = std::make_shared<Queue<shared_ptr<Batch>>>(pipeline_options_->gradients_host_queue_size);
        }
    }

    pipeline_lock_ = new std::mutex();
    max_batches_lock_ = new std::mutex();
    max_batches_cv_ = new std::condition_variable();

    staleness_bound_ = pipeline_options_->staleness_bound;
    batches_in_flight_ = 0;
    admitted_batches_ = 0;
    curr_pos_ = 0;


    fwd_round_robin_child_ = 0;
    fwd_round_robin_parent_ = 0;
    bwd_round_robin_child_ = 0;
//    bwd_round_robing_parent_ = 0;
    if (batch_worker_) {
        fwd_children_tags_ = vector<int>(model_->children_.size(), 100);
        bwd_children_tags_ = vector<int>(model_->children_.size(), 100);
    }
    if (compute_worker_) {
        fwd_parent_tags_ = vector<int>(model_->parents_.size(), 100);
        bwd_parent_tags_ = vector<int>(model_->parents_.size(), 100);
    }
    fwd_tag_lock_ = new std::mutex();
    bwd_tag_lock_ = new std::mutex();

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
    loaded_sliced_batches_ = nullptr;
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
            // TODO: fix number of threads assigned to each
            if (batch_worker_)
                addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);

            if (batch_worker_ and batch_worker_needs_remote_)
                addWorkersToPool(5, REMOTE_TO_DEVICE_ID, pipeline_options_->remote_transfer_threads);
            else if (compute_worker_) {
                addWorkersToPool(9, SLICE_BATCH_ID, pipeline_options_->batch_slice_threads);
                addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
            }

            if (compute_worker_ and compute_worker_needs_remote_)
                addWorkersToPool(6, REMOTE_LOADER_ID, pipeline_options_->remote_loader_threads);

            if (compute_worker_)
                addWorkersToPool(2, GPU_COMPUTE_ID, 1, model_->device_models_.size());  // Only one std::thread manages GPU

            if (model_->has_embeddings() and compute_worker_)
                addWorkersToPool(3, D2H_TRANSFER_ID, pipeline_options_->gradient_transfer_threads);

            if ((compute_worker_ and compute_worker_needs_remote_) or (batch_worker_ and batch_worker_needs_remote_))
                addWorkersToPool(8, REMOTE_TO_HOST_ID, pipeline_options_->remote_gradient_transfer_threads);
            else if (model_->has_embeddings() and batch_worker_)
                addWorkersToPool(4, UPDATE_BATCH_ID, pipeline_options_->gradient_update_threads);

            if (batch_worker_ and batch_worker_needs_remote_)
                addWorkersToPool(7, REMOTE_LISTEN_FOR_UPDATES_ID, pipeline_options_->remote_listen_threads);

        } else {
            if (batch_worker_)
                addWorkersToPool(0, LOAD_BATCH_ID, pipeline_options_->batch_loader_threads);

            if (batch_worker_ and batch_worker_needs_remote_)
                addWorkersToPool(5, REMOTE_TO_DEVICE_ID, pipeline_options_->batch_transfer_threads);
            else if (compute_worker_) {
                addWorkersToPool(9, SLICE_BATCH_ID, pipeline_options_->batch_slice_threads);
                addWorkersToPool(1, H2D_TRANSFER_ID, pipeline_options_->batch_transfer_threads);
            }

            if (compute_worker_ and compute_worker_needs_remote_)
                addWorkersToPool(6, REMOTE_LOADER_ID, pipeline_options_->batch_loader_threads);

            if (compute_worker_)
                addWorkersToPool(2, GPU_COMPUTE_ID, 1, model_->device_models_.size());  // Only one std::thread manages GPU

            if (compute_worker_ and compute_worker_needs_remote_)
                addWorkersToPool(3, D2H_TRANSFER_ID, pipeline_options_->gradient_transfer_threads);

            if ((compute_worker_ and compute_worker_needs_remote_) or (batch_worker_ and batch_worker_needs_remote_))
                addWorkersToPool(8, REMOTE_TO_HOST_ID, pipeline_options_->gradient_transfer_threads);

            if (batch_worker_ and batch_worker_needs_remote_)
                addWorkersToPool(7, REMOTE_LISTEN_FOR_UPDATES_ID, pipeline_options_->gradient_update_threads);
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
        loaded_sliced_batches_->flush();
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
        loaded_sliced_batches_->flush();
        for (auto d : device_loaded_batches_) {
            d->flush();
        }
    }
}

void PipelineGPU::setQueueExpectingData(bool expecting_data) {
    if (train_) {
        loaded_batches_->expecting_data_ = expecting_data;
        loaded_batches_->cv_->notify_all();
        loaded_sliced_batches_->expecting_data_ = expecting_data;
        loaded_sliced_batches_->cv_->notify_all();
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
        loaded_sliced_batches_->expecting_data_ = expecting_data;
        loaded_sliced_batches_->cv_->notify_all();
        for (auto d : device_loaded_batches_) {
            d->expecting_data_ = expecting_data;
            d->cv_->notify_all();
        }
    }
}
