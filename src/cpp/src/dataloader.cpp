//
// Created by Jason Mohoney on 2/7/20.
//

#include "dataloader.h"

#include "ordering.h"

DataLoader::DataLoader(GraphModelStorage *graph_storage,
                       shared_ptr<TrainingConfig> training_config,
                       shared_ptr<EvaluationConfig> evaluation_config,
                       shared_ptr<EncoderConfig> encoder_config) {

    current_edge_ = 0;
    train_ = true;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    graph_storage_ = graph_storage;
    training_config_ = training_config;
    evaluation_config_ = evaluation_config;

    edge_sampler_ = new RandomEdgeSampler(graph_storage_);

    if (graph_storage->learning_task_ == LearningTask::LINK_PREDICTION) {
        int num_uniform_train = training_config_->negative_sampling->negatives_per_positive *
                                (1 - training_config_->negative_sampling->degree_fraction);
        training_negative_sampler_ = new RandomNegativeSampler(graph_storage_,
                                                              training_config_->negative_sampling->num_chunks,
                                                              num_uniform_train);

        if (evaluation_config_->negative_sampling->filtered) {
            evaluation_negative_sampler_ = new FilteredNegativeSampler(graph_storage_);
        } else {
            int num_uniform_eval = evaluation_config_->negative_sampling->negatives_per_positive *
                                   (1 - evaluation_config_->negative_sampling->degree_fraction);
            evaluation_negative_sampler_ = new RandomNegativeSampler(graph_storage_,
                                                                     evaluation_config_->negative_sampling->num_chunks,
                                                                     num_uniform_eval);
        }
    } else {
        training_negative_sampler_ = nullptr;
        evaluation_negative_sampler_ = nullptr;
    }

    if (encoder_config != nullptr) {
        if (!encoder_config->layers.empty()) {
            std::vector<shared_ptr<NeighborSamplingConfig>> train_nbr_config;
            std::vector<shared_ptr<NeighborSamplingConfig>> eval_nbr_config;
            for (auto layer_config : encoder_config->layers) {
                train_nbr_config.emplace_back(layer_config->train_neighbor_sampling);
                if (layer_config->eval_neighbor_sampling != nullptr) {
                    eval_nbr_config.emplace_back(layer_config->eval_neighbor_sampling);
                } else {
                    eval_nbr_config.emplace_back(layer_config->train_neighbor_sampling);
                }

            }
            training_neighbor_sampler_ = new LayeredNeighborSampler(graph_storage_, train_nbr_config, encoder_config->use_incoming_nbrs, encoder_config->use_outgoing_nbrs);
            evaluation_neighbor_sampler_ = new LayeredNeighborSampler(graph_storage_, eval_nbr_config, encoder_config->use_incoming_nbrs, encoder_config->use_outgoing_nbrs);
        } else {
            training_neighbor_sampler_ = nullptr;
            evaluation_neighbor_sampler_ = nullptr;
        }
    } else {
        training_neighbor_sampler_ = nullptr;
        evaluation_neighbor_sampler_ = nullptr;
    }

}

DataLoader::~DataLoader() {

    delete sampler_lock_;
    delete edge_sampler_;

    delete batch_lock_;
    delete batch_cv_;

    if (training_negative_sampler_ != nullptr) {
        delete training_negative_sampler_;
    }

    if (evaluation_negative_sampler_ != nullptr) {
        if (training_negative_sampler_ != evaluation_negative_sampler_) {
            delete evaluation_negative_sampler_;
        }
    }

    if (training_neighbor_sampler_ != nullptr) {
        delete training_neighbor_sampler_;
    }

    if (evaluation_neighbor_sampler_ != nullptr) {
        if (training_neighbor_sampler_ != evaluation_neighbor_sampler_) {
            delete evaluation_neighbor_sampler_;
        }
    }
}

void DataLoader::nextEpoch() {
    batch_id_offset_ = 0;
    total_batches_processed_ = 0;
//    graph_storage_->shuffleEdges();
    epochs_processed_++;
}

void DataLoader::setActiveEdges() {
    bool set_active_edges = false;
    EdgeList active_edges;

    if (graph_storage_->useInMemorySubGraph()) {
        set_active_edges = true;

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
        int64_t total_size = local_offsets[-1].item<int64_t>();
        local_offsets = local_offsets - edge_bucket_sizes;

        auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

        active_edges = torch::empty({total_size, graph_storage_->storage_ptrs_.train_edges->dim1_size_}, graph_storage_->current_subgraph_state_->all_in_memory_mapped_edges_.options());

        #pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t idx = in_memory_edge_bucket_idx_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
            int64_t local_offset = local_offsets_accessor[i];

            active_edges.narrow(0, local_offset, edge_bucket_size) = graph_storage_->current_subgraph_state_->all_in_memory_mapped_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        }

    } else {
        if (train_) {
            set_active_edges = true;
            active_edges = graph_storage_->storage_ptrs_.train_edges->range(0, graph_storage_->storage_ptrs_.train_edges->getDim0());
        }
    }

    if (set_active_edges) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(active_edges.device());
        active_edges = (active_edges.index_select(0, torch::randperm(active_edges.size(0), opts)));
        graph_storage_->setActiveEdges(active_edges);
    }
}

void DataLoader::setActiveNodes() {
    bool set_active_nodes = false;
    torch::Tensor node_ids;

    if (graph_storage_->useInMemorySubGraph()) {
        set_active_nodes = true;
        node_ids = *node_ids_per_buffer_iterator_++;
    } else {
        if (train_) {
            set_active_nodes = true;
            node_ids = graph_storage_->storage_ptrs_.train_nodes->range(0, graph_storage_->storage_ptrs_.train_nodes->getDim0());
            node_ids = node_ids.flatten(0, 1);
        }
    }

    if (set_active_nodes) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
        node_ids = (node_ids.index_select(0, torch::randperm(node_ids.size(0), opts)));
        graph_storage_->setActiveNodes(node_ids);
    }
}

void DataLoader::initializeBatches() {
    int64_t batch_size = 0;
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    clearBatches();

    all_read_ = false;

    if (train_) {
        batch_size = training_config_->batch_size;
    } else {
        batch_size = evaluation_config_->batch_size;
    }
    int64_t num_items;

    if (graph_storage_->learning_task_ == LearningTask::LINK_PREDICTION) {
        setActiveEdges();
        num_items = graph_storage_->getNumActiveEdges();
    } else {
        setActiveNodes();
        num_items = graph_storage_->getNumActiveNodes();
    }


    vector<Batch *> batches;
    while(start_idx < num_items) {
        if (num_items - (start_idx + batch_size) < 0) {
            batch_size = num_items - start_idx;
        }
        Batch *curr_batch = new Batch(train_);
        curr_batch->batch_id_ = batch_id + batch_id_offset_;
        curr_batch->start_idx_ = start_idx;
        curr_batch->batch_size_ = batch_size;

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

    if (graph_storage_->storage_config_->embeddings != nullptr) {
        options = std::dynamic_pointer_cast<PartitionBufferOptions>(graph_storage_->storage_config_->embeddings->options);
    } else if (graph_storage_->storage_config_->features != nullptr) {
        options = std::dynamic_pointer_cast<PartitionBufferOptions>(graph_storage_->storage_config_->features->options);
    }

    if (graph_storage_->learning_task_ == LearningTask::LINK_PREDICTION) {
        if (graph_storage_->useInMemorySubGraph()) {

            auto tup = getEdgeBucketOrdering(options->edge_bucket_ordering,
                                             options->num_partitions,
                                             options->buffer_capacity,
                                             options->fine_to_coarse_ratio,
                                             options->num_cache_partitions,
                                             options->randomly_assign_edge_buckets);
            buffer_states_ = std::get<0>(tup);
            edge_buckets_per_buffer_ = std::get<1>(tup);

            edge_buckets_per_buffer_iterator_ = edge_buckets_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(options->node_partition_ordering,
                                                graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                                                graph_storage_->storage_config_->dataset->num_nodes,
                                                options->num_partitions,
                                                options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
        }
    }

}

void DataLoader::clearBatches() {
    for (Batch *b : batches_) {
        delete b;
    }
    batches_ = std::vector<Batch *>();
}

Batch *DataLoader::getNextBatch() {

    std::unique_lock batch_lock(*batch_lock_);
    batch_cv_->wait(batch_lock, [this] { return !waiting_for_batches_; });

    Batch *batch;
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


                graph_storage_->updateInMemorySubGraph();

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
    bool ret = !all_read_;
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


Batch *DataLoader::getBatch(int worker_id) {

    Batch *batch = getNextBatch();
    if (batch == nullptr) {
        return batch;
    }

    if (graph_storage_->learning_task_ == LearningTask::LINK_PREDICTION) {
        linkPredictionSample(batch, worker_id);
    } else if (graph_storage_->learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        nodeClassificationSample(batch, worker_id);
    }

    loadCPUParameters(batch);

    return batch;
}

std::vector<Batch *> DataLoader::getSubBatches() {

    Batch *batch = getNextBatch();

    if (batch == nullptr) {
        return {batch};
    }

    std::vector<Batch *> sub_batches;

    if (graph_storage_->learning_task_ == LearningTask::LINK_PREDICTION) {
        EdgeList pos_batch = edge_sampler_->getEdges(batch);
        int64_t num_edges = pos_batch.size(0);

        int num_devices = graph_storage_->storage_config_->device_ids.size();
        int64_t edges_per_batch = num_edges / num_devices;

        int64_t offset = 0;
        for (int i = 0; i < graph_storage_->storage_config_->device_ids.size(); i++) {

            Batch *sub_batch = new Batch(train_);

            if (offset + edges_per_batch > num_edges) {
                edges_per_batch = num_edges - offset;
            }

            sub_batch->batch_id_ = batch->batch_id_;
            sub_batch->start_idx_ = batch->start_idx_ + offset;
            sub_batch->batch_size_ = edges_per_batch;

            if (graph_storage_->storage_config_->dataset->num_relations == 1) {
                sub_batch->src_pos_indices_ = pos_batch.narrow(0, offset, edges_per_batch).select(1, 0);
                sub_batch->dst_pos_indices_ = pos_batch.narrow(0, offset, edges_per_batch).select(1, 1);
            } else {
                sub_batch->src_pos_indices_ = pos_batch.narrow(0, offset, edges_per_batch).select(1, 0);
                sub_batch->dst_pos_indices_ = pos_batch.narrow(0, offset, edges_per_batch).select(1, -1);
                sub_batch->rel_indices_ = pos_batch.narrow(0, offset, edges_per_batch).select(1, 1);
            }

            offset += edges_per_batch;

            linkPredictionSample(sub_batch);
            loadCPUParameters(sub_batch);
            sub_batches.emplace_back(sub_batch);
        }

    } else if (graph_storage_->learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        Indices node_ids = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
        int64_t num_nodes = node_ids.size(0);

        int num_devices = graph_storage_->storage_config_->device_ids.size();
        int64_t nodes_per_batch = num_nodes / num_devices;

        int64_t offset = 0;
        for (int i = 0; i < graph_storage_->storage_config_->device_ids.size(); i++) {

            Batch *sub_batch = new Batch(train_);
            if (offset + nodes_per_batch > num_nodes) {
                nodes_per_batch = num_nodes - offset;
            }

            sub_batch->batch_id_ = batch->batch_id_;
            sub_batch->start_idx_ = batch->start_idx_ + offset;
            sub_batch->batch_size_ = nodes_per_batch;

            sub_batch->root_node_indices_ = node_ids.narrow(0, offset, nodes_per_batch);
            offset += nodes_per_batch;

            nodeClassificationSample(sub_batch);
            loadCPUParameters(sub_batch);
            sub_batches.emplace_back(sub_batch);
        }
    }
    return sub_batches;
}

void DataLoader::linkPredictionSample(Batch *batch, int worker_id) {

    if (!batch->src_pos_indices_.defined()) {
        EdgeList pos_batch = edge_sampler_->getEdges(batch);
        if (graph_storage_->storage_config_->dataset->num_relations == 1) {
            batch->src_pos_indices_ = pos_batch.select(1, 0);
            batch->dst_pos_indices_ = pos_batch.select(1, 1);
        } else {
            batch->src_pos_indices_ = pos_batch.select(1, 0);
            batch->dst_pos_indices_ = pos_batch.select(1, -1);
            batch->rel_indices_ = pos_batch.select(1, 1);
        }
    }

    if (train_) {
        batch->negative_sampling_ = training_config_->negative_sampling;
    } else {
        batch->negative_sampling_ = evaluation_config_->negative_sampling;
    }

    if (batch->negative_sampling_->filtered) {
        batch->negative_sampling_->negatives_per_positive = graph_storage_->getNumNodesInMemory();
    }

    negative_sampler_->lock();
    // TODO only grab these if the decoder options have inverse edges set to true
    batch->src_neg_indices_ = negative_sampler_->getNegatives(batch, true);

    batch->dst_neg_indices_ = negative_sampler_->getNegatives(batch, false);
    negative_sampler_->unlock();

    if (neighbor_sampler_ != nullptr) {
        // get unique nodes of the edges and negatives in the batch

        batch->setUniqueNodes(false, false);

        batch->gnn_graph_ = neighbor_sampler_->getNeighbors(batch->unique_node_indices_, worker_id);

        // update the mapping with the neighbors
        batch->setUniqueNodes(true, true);
    } else {
        batch->setUniqueNodes(false, true);
    }

    if (!train_ && graph_storage_->filtered_eval_) {
        graph_storage_->setEvalFilter(batch);
    }
}

void DataLoader::nodeClassificationSample(Batch *batch, int worker_id) {

    if (!batch->root_node_indices_.defined()) {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
        batch->unique_node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_);
        if (graph_storage_->current_subgraph_state_->global_to_local_index_map_.defined()) {
            batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
        }
    }

    batch->gnn_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, worker_id);

    // update the mapping with the neighbors
    batch->unique_node_indices_ = batch->gnn_graph_.getNodeIDs();
}

void DataLoader::loadCPUParameters(Batch *batch) {

    if (graph_storage_->storage_config_->embeddings != nullptr) {
        if (graph_storage_->storage_config_->embeddings->type != StorageBackend::DEVICE_MEMORY) {
            batch->unique_node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->unique_node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
            }
        }
    }

    if (graph_storage_->storage_config_->features != nullptr) {
        if (graph_storage_->storage_config_->features->type != StorageBackend::DEVICE_MEMORY) {
            batch->unique_node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
        }
    }

    batch->status_ = BatchStatus::LoadedEmbeddings;
    batch->load_timestamp_ = timestamp_;
}

void DataLoader::loadGPUParameters(Batch *batch) {
    if (graph_storage_->storage_config_->embeddings != nullptr) {
        if (graph_storage_->storage_config_->embeddings->type == StorageBackend::DEVICE_MEMORY) {
            batch->unique_node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->unique_node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
            }
        }
    }

    if (graph_storage_->storage_config_->features != nullptr) {
        if (graph_storage_->storage_config_->features->type == StorageBackend::DEVICE_MEMORY) {
            batch->unique_node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            // batch->unique_node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_); already set
        }
    }
}

void DataLoader::updateEmbeddingsForBatch(Batch *batch, bool gpu) {
    // TODO: if this method is called without embeddings (i.e. in node classification it may fail)
    if (gpu) {
        if (graph_storage_->storage_config_->embeddings->type == StorageBackend::DEVICE_MEMORY) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->unique_node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->unique_node_state_update_);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_config_->embeddings->type != StorageBackend::DEVICE_MEMORY ||
                (graph_storage_->storage_config_->embeddings->type == StorageBackend::DEVICE_MEMORY &&
                graph_storage_->storage_config_->device_type == torch::kCPU)) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->unique_node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->unique_node_state_update_);
        }
        batch->clear();
    }
}

void DataLoader::loadStorage() {
    setBufferOrdering();
    graph_storage_->load();

    batch_id_offset_ = 0;
    batches_left_ = 0;
    total_batches_processed_ = 0;
    all_read_ = false;

    int num_hash_maps = 0;
    if (train_) {
        if (training_config_->pipeline->sync) {
            num_hash_maps = 1;
        } else {
            num_hash_maps = training_config_->pipeline->batch_loader_threads;
        }
    } else {
        if (evaluation_config_->pipeline->sync) {
            num_hash_maps = 1;
        } else {
            num_hash_maps = evaluation_config_->pipeline->batch_loader_threads;
        }
    }

    if (!buffer_states_.empty()) {
        graph_storage_->initializeInMemorySubGraph(buffer_states_[0], num_hash_maps);
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}), num_hash_maps);
    }

    initializeBatches();
}