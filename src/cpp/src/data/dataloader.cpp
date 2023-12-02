//
// Created by Jason Mohoney on 2/7/20.
//

#include "data/dataloader.h"

#include "common/util.h"
#include "data/ordering.h"

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, bool use_partition_embeddings,
                       shared_ptr<TrainingConfig> training_config, shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config,
                       bool batch_worker, bool compute_worker) {
    current_edge_ = 0;
    train_ = true;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    single_dataset_ = false;

    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    training_config_ = training_config;
    evaluation_config_ = evaluation_config;
    only_root_features_ = false;

    use_partition_embeddings_ = use_partition_embeddings;
    batch_worker_ = batch_worker;
    compute_worker_ = compute_worker;

    if (!batch_worker) {

    } else {
        edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);

        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            training_negative_sampler_ = std::make_shared<CorruptNodeNegativeSampler>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->local_filter_mode);

            evaluation_negative_sampler_ = std::make_shared<CorruptNodeNegativeSampler>(
                evaluation_config_->negative_sampling->num_chunks, evaluation_config_->negative_sampling->negatives_per_positive,
                evaluation_config_->negative_sampling->degree_fraction, evaluation_config_->negative_sampling->filtered,
                evaluation_config_->negative_sampling->local_filter_mode);
        } else {
            training_negative_sampler_ = nullptr;
            evaluation_negative_sampler_ = nullptr;
        }

        if (encoder_config != nullptr) {
            if (!encoder_config->train_neighbor_sampling.empty()) {
                training_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->train_neighbor_sampling,
                                                                                      encoder_config->use_incoming_nbrs, encoder_config->use_outgoing_nbrs);

                if (!encoder_config->eval_neighbor_sampling.empty()) {
                    evaluation_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->eval_neighbor_sampling,
                                                                                            encoder_config->use_incoming_nbrs, encoder_config->use_incoming_nbrs);
                } else {
                    evaluation_neighbor_sampler_ = training_neighbor_sampler_;
                }

            } else {
                training_neighbor_sampler_ = nullptr;
                evaluation_neighbor_sampler_ = nullptr;
            }
        } else {
            training_neighbor_sampler_ = nullptr;
            evaluation_neighbor_sampler_ = nullptr;
        }
    }

    compute_streams_ = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // TODO: fix this
//    for (int i = 0; i < graph_storage_->num_gpus_; i++) {
//        compute_streams_.emplace_back(nullptr);
//    }

//    compute_streams_(graph_storage_->num_gpus_); //= {nullptr, nullptr};
//    for (int i = 0; i < graph_storage_->num_gpus_; i++) {
//        compute_streams_[i] = nullptr;
//    }

//    CudaStream* temp[graph_storage_->num_gpus_];
//    for (int i = 0; i < graph_storage_->num_gpus_; i++) {
//        temp[i] = nullptr;
//    }
//    compute_streams_ = &temp[0];

    pg_gloo_ = nullptr;
    dist_config_ = nullptr;
//    dist_ = false;


    int num_hash_maps = training_config_->pipeline->batch_slice_threads;
    if (training_config_->pipeline->remote_transfer_threads > num_hash_maps) num_hash_maps = training_config_->pipeline->remote_transfer_threads;
    if (num_hash_maps > 0) {
        auto bool_device_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
        for (int i = 0; i < num_hash_maps; i++) {
            hash_maps_.emplace_back(torch::zeros({graph_storage_->getNumNodes()}, bool_device_options));
        }
    }
}

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler,
                       shared_ptr<NeighborSampler> neighbor_sampler, bool train) {
    current_edge_ = 0;
    train_ = train;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    batch_size_ = batch_size;
    single_dataset_ = true;

    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);
    negative_sampler_ = negative_sampler;
    neighbor_sampler_ = neighbor_sampler;

    training_config_ = nullptr;
    evaluation_config_ = nullptr;

    training_negative_sampler_ = nullptr;
    evaluation_negative_sampler_ = nullptr;

    training_neighbor_sampler_ = nullptr;
    evaluation_neighbor_sampler_ = nullptr;

    loadStorage();
}

DataLoader::~DataLoader() {
    delete sampler_lock_;
    delete batch_lock_;
    delete batch_cv_;
}

void DataLoader::nextEpoch() {
    batch_id_offset_ = 0;
    total_batches_processed_ = 0;
    epochs_processed_++;

    if (graph_storage_->useInMemorySubGraph()) {
        unloadStorage(true);
    }
}

void DataLoader::setActiveEdges() {
    EdgeList active_edges;

    if (graph_storage_->useInMemorySubGraph()) {
        if (neighbor_sampler_ == nullptr) {
            graph_storage_->setActiveEdges(graph_storage_->current_subgraph_state_->all_in_memory_mapped_edges_);
            return;
        }

        torch::Tensor edge_bucket_ids = *edge_buckets_per_buffer_iterator_;
        edge_buckets_per_buffer_iterator_++;

        int num_partitions = graph_storage_->getNumPartitions();

        edge_bucket_ids = edge_bucket_ids.select(1, 0) * num_partitions + edge_bucket_ids.select(1, 1);
        torch::Tensor in_memory_edge_bucket_idx = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());
        torch::Tensor edge_bucket_sizes = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());

        auto edge_bucket_ids_accessor = edge_bucket_ids.accessor<int64_t, 1>();
        auto in_memory_edge_bucket_idx_accessor = in_memory_edge_bucket_idx.accessor<int64_t, 1>();
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();

        auto all_edge_bucket_sizes_accessor = graph_storage_->current_subgraph_state_->in_memory_edge_bucket_sizes_.accessor<int64_t, 1>();
        auto all_edge_bucket_starts_accessor = graph_storage_->current_subgraph_state_->in_memory_edge_bucket_starts_.accessor<int64_t, 1>();

        auto tup = torch::sort(graph_storage_->current_subgraph_state_->in_memory_edge_bucket_ids_);
        torch::Tensor sorted_in_memory_ids = std::get<0>(tup);
        torch::Tensor in_memory_id_indices = std::get<1>(tup);
        auto in_memory_id_indices_accessor = in_memory_id_indices.accessor<int64_t, 1>();

        #pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t edge_bucket_id = edge_bucket_ids_accessor[i];
            int64_t idx = torch::searchsorted(sorted_in_memory_ids, edge_bucket_id).item<int64_t>();
            idx = in_memory_id_indices_accessor[idx];
            int64_t edge_bucket_size = all_edge_bucket_sizes_accessor[idx];

            in_memory_edge_bucket_idx_accessor[i] = idx;
            edge_bucket_sizes_accessor[i] = edge_bucket_size;
        }

        torch::Tensor local_offsets = edge_bucket_sizes.cumsum(0);
        int64_t total_size = 0;
        if (local_offsets.size(0) > 0) total_size = local_offsets[-1].item<int64_t>();
        local_offsets = local_offsets - edge_bucket_sizes;

        auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

        active_edges = torch::empty({total_size, graph_storage_->storage_ptrs_.edges->dim1_size_},
                                    graph_storage_->current_subgraph_state_->all_in_memory_mapped_edges_.options());

        #pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t idx = in_memory_edge_bucket_idx_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
            int64_t local_offset = local_offsets_accessor[i];

            active_edges.narrow(0, local_offset, edge_bucket_size) =
                graph_storage_->current_subgraph_state_->all_in_memory_mapped_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        }

    } else {
        active_edges = graph_storage_->storage_ptrs_.edges->range(0, graph_storage_->storage_ptrs_.edges->getDim0());
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(active_edges.device());
    active_edges = (active_edges.index_select(0, torch::randperm(active_edges.size(0), opts)));
    graph_storage_->setActiveEdges(active_edges);
}

void DataLoader::setActiveNodes() {
    torch::Tensor node_ids;

    if (graph_storage_->useInMemorySubGraph()) {
        node_ids = *node_ids_per_buffer_iterator_++;
    } else {
        node_ids = graph_storage_->storage_ptrs_.nodes->range(0, graph_storage_->storage_ptrs_.nodes->getDim0());
        if (node_ids.sizes().size() == 2) {
            node_ids = node_ids.flatten(0, 1);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    node_ids = (node_ids.index_select(0, torch::randperm(node_ids.size(0), opts)));
    graph_storage_->setActiveNodes(node_ids);
}

void DataLoader::initializeBatches(bool prepare_encode) {
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    clearBatches();

    all_read_ = false;
    int64_t num_items;

    if (prepare_encode) {
        num_items = graph_storage_->getNumNodes();
    } else {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setActiveEdges();
            num_items = graph_storage_->getNumActiveEdges();
        } else {
            setActiveNodes();
            num_items = graph_storage_->getNumActiveNodes();
        }
    }

    int64_t batch_size = batch_size_;
    vector<shared_ptr<Batch>> batches;
    while (start_idx < num_items) {
        if (num_items - (start_idx + batch_size) < 0) {
            batch_size = num_items - start_idx;
        }
        shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train_);
        curr_batch->batch_id_ = batch_id + batch_id_offset_;
        curr_batch->start_idx_ = start_idx;
        curr_batch->batch_size_ = batch_size;
        curr_batch->num_sub_batches_ = num_sub_batches_;

        if (prepare_encode) {
            curr_batch->task_ = LearningTask::ENCODE;
        } else {
            curr_batch->task_ = learning_task_;
        }

        batches.emplace_back(curr_batch);
        batch_id++;
        start_idx += batch_size;
    }
    batches_ = batches;

    batches_left_ = batches_.size();
    batch_iterator_ = batches_.begin();
}

void DataLoader::setBufferOrdering() {
    shared_ptr<PartitionBufferOptions> options;

    if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)->options_;
    }

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (graph_storage_->useInMemorySubGraph()) {
            auto tup = getEdgeBucketOrdering(options->edge_bucket_ordering, options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio,
                                             options->num_cache_partitions, options->randomly_assign_edge_buckets, pg_gloo_, dist_config_);
            buffer_states_ = std::get<0>(tup);
            edge_buckets_per_buffer_ = std::get<1>(tup);

            edge_buckets_per_buffer_iterator_ = edge_buckets_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_, edge_buckets_per_buffer_iterator_);

            // TODO: calculate num examples here (and likewise for below),
            //  progress reporter would need to be updated each epoch based on this in the trainer
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(
                options->node_partition_ordering, graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                graph_storage_->getNumNodes(), options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions, pg_gloo_, dist_config_);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_, node_ids_per_buffer_iterator_);
        }
    }

}

void DataLoader::clearBatches() { batches_ = std::vector<shared_ptr<Batch>>(); }

shared_ptr<Batch> DataLoader::getNextBatch() {
    std::unique_lock batch_lock(*batch_lock_);
    batch_cv_->wait(batch_lock, [this] { return !waiting_for_batches_; });

    shared_ptr<Batch> batch;
    if (batch_iterator_ != batches_.end()) {
        batch = *batch_iterator_;
        batch_iterator_++;

        // check if all batches have been read
        if (batch_iterator_ == batches_.end()) {
            if (graph_storage_->useInMemorySubGraph()) {
                if (!graph_storage_->hasSwap()) {
                    all_read_ = true;
                }
            } else {
                all_read_ = true;
            }
        }
    } else {
        batch = nullptr;
        if (graph_storage_->useInMemorySubGraph()) {
            if (graph_storage_->hasSwap()) {
                // wait for all batches to finish before swapping
                waiting_for_batches_ = true;
                batch_cv_->wait(batch_lock, [this] { return batches_left_ == 0; });
                waiting_for_batches_ = false;

                graph_storage_->updateInMemorySubGraph(neighbor_sampler_ != nullptr);

                initializeBatches();
                batch = *batch_iterator_;
                batch_iterator_++;

                // check if all batches have been read
                if (batch_iterator_ == batches_.end()) {
                    if (graph_storage_->useInMemorySubGraph()) {
                        if (!graph_storage_->hasSwap()) {
                            all_read_ = true;
                        }
                    } else {
                        all_read_ = true;
                    }
                }
            } else {
                all_read_ = true;
            }
        } else {
            all_read_ = true;
        }
    }
    batch_lock.unlock();
    batch_cv_->notify_all();
    return batch;
}

bool DataLoader::hasNextBatch() {
    batch_lock_->lock();
    bool ret = !all_read_ && !(batches_.size() == 0 and graph_storage_->useInMemorySubGraph() and !graph_storage_->hasSwap());
    batch_lock_->unlock();
    return ret;
}

void DataLoader::finishedBatch() {
    batch_lock_->lock();
    batches_left_--;
    total_batches_processed_++;
    batch_lock_->unlock();
    batch_cv_->notify_all();
}

void DataLoader::getBatchHelper(shared_ptr<Batch> batch, int worker_id) {
    if (batch->task_ == LearningTask::LINK_PREDICTION) {
        edgeSample(batch, worker_id);
    } else if (batch->task_ == LearningTask::NODE_CLASSIFICATION || batch->task_ == LearningTask::ENCODE) {
        nodeSample(batch, worker_id);
    }

    if (use_partition_embeddings_) { // TODO: maybe check that we are actually using the buffer during training here
        if (!batch->dense_graph_.node_ids_.defined()) {
            //need to set
            batch->dense_graph_.node_ids_ = batch->unique_node_indices_;
        }
        if (graph_storage_->useInMemorySubGraph()) {
            batch->dense_graph_.buffer_state_ = graph_storage_->getBufferState();
        }
        batch->dense_graph_.partition_size_ = graph_storage_->getPartitionSize();
    }

//    loadCPUParameters(batch);
}

shared_ptr<Batch> DataLoader::getBatch(at::optional<torch::Device> device, bool perform_map, int worker_id) {
    shared_ptr<Batch> batch = getNextBatch();
    if (batch == nullptr) {
        return batch;
    }


    if (train_ and graph_storage_->num_gpus_ > 1) {
        std::vector <shared_ptr<Batch>> sub_batches;
        int num_batches = graph_storage_->num_gpus_ * num_sub_batches_;        //graph_storage_->storage_config_->device_ids.size();
//        std::cout<<"num_batches: "<<num_batches<<"\n";

        if (batch->task_ == LearningTask::LINK_PREDICTION) {
            torch::Tensor edges = edge_sampler_->getEdges(batch);
            int64_t num_edges = edges.size(0);

            int64_t edges_per_batch = num_edges / num_batches;

            int64_t offset = 0;
            for (int i = 0; i < num_batches; i++) {
                shared_ptr<Batch> sub_batch = std::make_shared<Batch>(train_);
                if (offset + edges_per_batch > num_edges) {
                    edges_per_batch = num_edges - offset;
                }

                sub_batch->task_ = batch->task_;
                sub_batch->batch_id_ = batch->batch_id_;
                sub_batch->start_idx_ = batch->start_idx_ + offset;
                sub_batch->batch_size_ = edges_per_batch;

//                sub_batch->root_node_indices_ = node_ids.narrow(0, offset, nodes_per_batch);
                offset += edges_per_batch;

                getBatchHelper(sub_batch, worker_id); // TODO: could put this in an omp for loop
                sub_batches.emplace_back(sub_batch);
            }


        } else {
            torch::Tensor node_ids = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(
                    torch::kInt64);
            int64_t num_nodes = node_ids.size(0);

            int64_t nodes_per_batch = num_nodes / num_batches;

            int64_t offset = 0;
            for (int i = 0; i < num_batches; i++) {
                shared_ptr<Batch> sub_batch = std::make_shared<Batch>(train_);
                if (offset + nodes_per_batch > num_nodes) {
                    nodes_per_batch = num_nodes - offset;
                }

                sub_batch->task_ = batch->task_;
                sub_batch->batch_id_ = batch->batch_id_;
                sub_batch->start_idx_ = batch->start_idx_ + offset;
                sub_batch->batch_size_ = nodes_per_batch;

//                sub_batch->root_node_indices_ = node_ids.narrow(0, offset, nodes_per_batch);
                offset += nodes_per_batch;

                getBatchHelper(sub_batch, worker_id);
                sub_batches.emplace_back(sub_batch);
            }
        }

        batch->sub_batches_ = sub_batches;
//        std::cout<<"batch sub batches: "<<batch->sub_batches_.size()<<"\n";

//        if (compute_worker_)
//            loadCPUParameters(batch, worker_id);

        return batch;
    }

    // single GPU
    getBatchHelper(batch, worker_id);

    if (device.has_value()) {
        if (device.value().is_cuda()) {
            batch->to(device.value());
            loadGPUParameters(batch);
            //            batch->dense_graph_.performMap();
        }
    }

    if (perform_map) {
        batch->dense_graph_.performMap();
    }

    return batch;
}

void DataLoader::edgeSample(shared_ptr<Batch> batch, int worker_id) {
    if (!batch->edges_.defined()) {
        batch->edges_ = edge_sampler_->getEdges(batch);
    }

    if (negative_sampler_ != nullptr) {
        negativeSample(batch);
    }

    std::vector<torch::Tensor> all_ids = {batch->edges_.select(1, 0), batch->edges_.select(1, -1)};

    if (batch->src_neg_indices_.defined()) {
        all_ids.emplace_back(batch->src_neg_indices_.flatten(0, 1));
    }

    if (batch->dst_neg_indices_.defined()) {
        all_ids.emplace_back(batch->dst_neg_indices_.flatten(0, 1));
    }

    torch::Tensor src_mapping;
    torch::Tensor dst_mapping;
    torch::Tensor src_neg_mapping;
    torch::Tensor dst_neg_mapping;

    std::vector<torch::Tensor> mapped_tensors;

    if (neighbor_sampler_ != nullptr) {
        // get unique nodes in edges and negatives
        batch->root_node_indices_ = std::get<0>(torch::_unique(torch::cat(all_ids)));

        // sample neighbors and get unique nodes
        batch->dense_graph_ =
            neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_, worker_id);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();

        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = torch::sort(batch->unique_node_indices_);
        torch::Tensor sorted_map = std::get<0>(tup);
        torch::Tensor map_to_unsorted = std::get<1>(tup);

        mapped_tensors = apply_tensor_map(sorted_map, all_ids);

        int64_t num_nbrs_sampled = batch->dense_graph_.hop_offsets_[-2].item<int64_t>();

        src_mapping = map_to_unsorted.index_select(0, mapped_tensors[0]) - num_nbrs_sampled;
        dst_mapping = map_to_unsorted.index_select(0, mapped_tensors[1]) - num_nbrs_sampled;

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[2]).reshape(batch->src_neg_indices_.sizes()) - num_nbrs_sampled;
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[3]).reshape(batch->dst_neg_indices_.sizes()) - num_nbrs_sampled;
        }
    } else {
        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = map_tensors(all_ids);
        batch->unique_node_indices_ = std::get<0>(tup);
        mapped_tensors = std::get<1>(tup);

        src_mapping = mapped_tensors[0];
        dst_mapping = mapped_tensors[1];

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = mapped_tensors[2].reshape(batch->src_neg_indices_.sizes());
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = mapped_tensors[3].reshape(batch->dst_neg_indices_.sizes());
        }
    }

    if (batch->edges_.size(1) == 2) {
        batch->edges_ = torch::stack({src_mapping, dst_mapping}).transpose(0, 1);
    } else if (batch->edges_.size(1) == 3) {
        batch->edges_ = torch::stack({src_mapping, batch->edges_.select(1, 1), dst_mapping}).transpose(0, 1);
    } else {
        throw TensorSizeMismatchException(batch->edges_, "Edge list must be a 3 or 2 column tensor");
    }

    batch->src_neg_indices_mapping_ = src_neg_mapping;
    batch->dst_neg_indices_mapping_ = dst_neg_mapping;
}

void DataLoader::nodeSample(shared_ptr<Batch> batch, int worker_id) {
    if (batch->task_ == LearningTask::ENCODE) {
        torch::TensorOptions node_opts = torch::TensorOptions().dtype(torch::kInt64).device(graph_storage_->storage_ptrs_.edges->device_);
        batch->root_node_indices_ = torch::arange(batch->start_idx_, batch->start_idx_ + batch->batch_size_, node_opts);
    } else {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
    }

    if (graph_storage_->storage_ptrs_.node_labels != nullptr) {
        batch->node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_).flatten(0, 1);
    }

    if (graph_storage_->useInMemorySubGraph()) {
//        batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
//        auto root_node_accessor = batch->root_node_indices_.accessor<int64_t, 1>();
//        auto buffer_offsets_accessor = graph_storage_->current_subgraph_state_->buffer_offsets_.accessor<int64_t, 1>();
//        int64_t partition_size = graph_storage_->getPartitionSize();
//
//        #pragma omp parallel for
//        for (int i = 0; i < batch->root_node_indices_.size(0); i++) {
//            int64_t global_id = root_node_accessor[i];
//            int64_t partition = global_id / partition_size;
//
//            root_node_accessor[i] = buffer_offsets_accessor[partition] + global_id - (partition * partition_size);
//        }

    }

    if (neighbor_sampler_ != nullptr) {
        batch->dense_graph_ =
            neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_, worker_id);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();
    } else {
        batch->unique_node_indices_ = batch->root_node_indices_;
    }
}

void DataLoader::negativeSample(shared_ptr<Batch> batch) {
    std::tie(batch->src_neg_indices_, batch->src_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_state_->in_memory_subgraph_, batch->edges_, true);
    std::tie(batch->dst_neg_indices_, batch->dst_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_state_->in_memory_subgraph_, batch->edges_, false);
}

void DataLoader::loadCPUParameters(shared_ptr<Batch> batch, int id, bool load) {
//    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
//        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
//            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
//            if (train_) {
//                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
//            }
//        }
//    }

//    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
//    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
//        if (graph_storage_->storage_ptrs_.node_features->device_ != torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
//                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);



                if (batch->sub_batches_.size() > 0) {

                    if (batch->sub_batches_[0]->node_features_.defined()) {
//                        std::cout<<"ALREADY LOADED\n";
                        return;
                    }

                    torch::Tensor unique_indices;

                    if (batch->creator_id_ != -1) {
                        // received this batch, already have the uniques on the root node_indices_
//                        std::cout<<"received completed batch\n";
//                        unique_indices = batch->sub_batches_[0]->root_node_indices_;
                    } else {
//                        std::vector <torch::Tensor> all_unique_nodes_vec(batch->sub_batches_.size());
//
//                        for (int i = 0; i < batch->sub_batches_.size(); i++) {
//                            all_unique_nodes_vec[i] = batch->sub_batches_[i]->unique_node_indices_;
//                        }
//
////                        Timer t = new Timer(false);
////                        t.start();
//                        torch::Tensor all_unique_nodes = torch::cat({all_unique_nodes_vec}, 0);
//                        unique_indices = computeUniques(all_unique_nodes, graph_storage_->getNumNodesInMemory(), id);
//
//                        batch->sub_batches_[0]->root_node_indices_ = unique_indices;
//                        for (int i = 1; i < batch->sub_batches_.size(); i++) {
//                            batch->sub_batches_[i]->root_node_indices_ = torch::Tensor();
//                        }
//
////                        t.stop();
////                        std::cout<< "calculated and set uniques: " << t.getDuration() << "\n";
////                        std::cout<<unique_indices.size(0)<<" vs "<<all_unique_nodes.size(0)<<"\n";
                    }

                    if (load) {
//                        std::cout<<"load\n";
//                        torch::Tensor unique_features = graph_storage_->getNodeFeatures(unique_indices);

//                        int split_size = (int) ceil((float) unique_indices.size(0) / batch->sub_batches_.size());
//                        int padded_size = split_size*batch->sub_batches_.size();
//                        int actual_size = unique_indices.size(0);
//                        unique_indices = torch::cat({unique_indices, torch::zeros({padded_size-actual_size}, unique_indices.options())}, 0);

                        #pragma omp parallel for
                        for (int i = 0; i < batch->sub_batches_.size(); i++) {
//                            int size = split_size;
//                            int start = i*split_size;
//
//                            if (start + size > unique_indices.size(0)) {
//                                size = unique_indices.size(0) - start;
//                            }

//                            batch->sub_batches_[i]->root_node_indices_ = unique_features.narrow(0, count, size);


                            if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
//                                batch->sub_batches_[i]->node_embeddings_ = graph_storage_->getNodeEmbeddings(unique_indices.narrow(0, start, size));
                                batch->sub_batches_[i]->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->sub_batches_[i]->unique_node_indices_);
                                if (train_) {
//                                    batch->sub_batches_[i]->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(unique_indices.narrow(0, start, size));
                                    batch->sub_batches_[i]->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->sub_batches_[i]->unique_node_indices_);
                                }
                            } else {
//                                batch->sub_batches_[i]->node_features_ = graph_storage_->getNodeFeatures(unique_indices.narrow(0, start, size));
                                batch->sub_batches_[i]->node_features_ = graph_storage_->getNodeFeatures(batch->sub_batches_[i]->unique_node_indices_);
                            }


                        }
                    }


                } else {
                    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
                        if (!batch->node_embeddings_.defined()) {
                            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
                            if (train_) {
                                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
                            }
                        }
                    } else {
                        if (!batch->node_features_.defined())
                            batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
                    }
                }




            }
//        }
//    }

    batch->status_ = BatchStatus::LoadedEmbeddings;
    batch->load_timestamp_ = timestamp_;
}

void DataLoader::loadGPUParameters(shared_ptr<Batch> batch) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ == torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }
}

void DataLoader::updateEmbeddings(shared_ptr<Batch> batch, bool gpu) {
    if (batch->sub_batches_.size() > 0) {
        #pragma omp parallel for // TODO: maybe not parallel for better perf?
        for (int i = 0; i < batch->sub_batches_.size(); i++) {
            if (batch->sub_batches_[i]->node_gradients_.defined()) {
                updateEmbeddings(batch->sub_batches_[i], gpu);
            }
        }
        return;
    }

    if (batch->node_gradients_.defined()) {
        if (gpu) {
            if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
                graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
                graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
            }
        } else {
//            batch->host_transfer_.synchronize();
            if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
                graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
                graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
            }
//            batch->clear();
        }
    }
}

void DataLoader::loadStorage() {
    setBufferOrdering();
    graph_storage_->load();

    batch_id_offset_ = 0;
    batches_left_ = 0;
    total_batches_processed_ = 0;
    all_read_ = false;

    int num_hash_maps = 1;
    if (train_) {
        if (training_config_ != nullptr && !training_config_->pipeline->sync) {
            num_hash_maps = training_config_->pipeline->batch_loader_threads;
        }
    } else {
        if (evaluation_config_ != nullptr && !evaluation_config_->pipeline->sync) {
            num_hash_maps = evaluation_config_->pipeline->batch_loader_threads;
        }
    }

    if (!buffer_states_.empty()) {
        graph_storage_->initializeInMemorySubGraph(buffer_states_[0], num_hash_maps, neighbor_sampler_ != nullptr);
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}), num_hash_maps, neighbor_sampler_ != nullptr);
    }

    if (negative_sampler_ != nullptr) {
        if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler_)) {
            if (std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler_)->filtered_) {
                graph_storage_->sortAllEdges();
            }
        }
    }
}

torch::Tensor DataLoader::computeUniques(torch::Tensor node_ids, int64_t num_nodes_in_memory, int id) {
    unsigned int num_threads = 1;

    #ifdef MARIUS_OMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    #endif

    int64_t chunk_size = ceil((double)num_nodes_in_memory / num_threads);

//    auto bool_device_options = torch::TensorOptions().dtype(torch::kBool).device(node_ids.device());
//    torch::Tensor hash_map = torch::zeros({num_nodes_in_memory}, bool_device_options);
    torch::Tensor hash_map = hash_maps_[id];

    auto hash_map_accessor = hash_map.accessor<bool, 1>();
    auto nodes_accessor = node_ids.accessor<int64_t, 1>();

    #pragma omp parallel default(none) shared(hash_map_accessor, hash_map, node_ids, nodes_accessor) num_threads(num_threads)
    {

        #pragma omp for
        for (int64_t j = 0; j < node_ids.size(0); j++) {
            if (!hash_map_accessor[nodes_accessor[j]]) {
                hash_map_accessor[nodes_accessor[j]] = 1;
            }
        }
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    std::vector<torch::Tensor> sub_deltas = std::vector<torch::Tensor>(num_threads);
    int64_t upper_bound = (int64_t)(node_ids.size(0)) / num_threads + 1;

    std::vector<int> sub_counts = std::vector<int>(num_threads, 0);
    std::vector<int> sub_offsets = std::vector<int>(num_threads, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        #ifdef MARIUS_OMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

//        if (tid == 30)
//            std::cout<<"omp: "<<tid<<"\n";

        sub_deltas[tid] = torch::empty({upper_bound}, device_options);
        auto delta_ids_accessor = sub_deltas[tid].accessor<int64_t, 1>();

        int64_t start = chunk_size * tid;
        int64_t end = start + chunk_size;

        if (end > num_nodes_in_memory) {
            end = num_nodes_in_memory;
        }

        int private_count = 0;
        int grow_count = 0;

        #pragma unroll
        for (int64_t j = start; j < end; j++) {
            if (hash_map_accessor[j]) {
                delta_ids_accessor[private_count++] = j;
                hash_map_accessor[j] = 0;
                grow_count++;

                if (grow_count == upper_bound) {
                    sub_deltas[tid] = torch::cat({sub_deltas[tid], torch::empty({upper_bound}, device_options)}, 0);
                    delta_ids_accessor = sub_deltas[tid].accessor<int64_t, 1>();
                    grow_count = 0;
                }
            }
        }
        sub_counts[tid] = private_count;
    }

    int count = 0;
    for (auto c : sub_counts) {
        count += c;
    }

    for (int k = 0; k < num_threads - 1; k++) {
        sub_offsets[k + 1] = sub_offsets[k] + sub_counts[k];
    }

    torch::Tensor delta_ids = torch::empty({count}, device_options);

    #pragma omp parallel for num_threads(num_threads)
    for (int k = 0; k < num_threads; k++) {
        if (sub_deltas[k].size(0) > 0)
            delta_ids.narrow(0, sub_offsets[k], sub_counts[k]) = sub_deltas[k].narrow(0, 0, sub_counts[k]);
    }

    return delta_ids;
}