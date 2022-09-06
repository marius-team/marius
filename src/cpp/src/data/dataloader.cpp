//
// Created by Jason Mohoney on 2/7/20.
//

#include "data/dataloader.h"

#include "common/util.h"
#include "data/ordering.h"

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
                       shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config) {
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
            training_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->train_neighbor_sampling);

            if (!encoder_config->eval_neighbor_sampling.empty()) {
                evaluation_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->eval_neighbor_sampling);
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
        unloadStorage();
    }
}

void DataLoader::setActiveEdges() {
    EdgeList active_edges;

    if (graph_storage_->useInMemorySubGraph()) {
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
                                             options->num_cache_partitions, options->randomly_assign_edge_buckets);
            buffer_states_ = std::get<0>(tup);
            edge_buckets_per_buffer_ = std::get<1>(tup);

            edge_buckets_per_buffer_iterator_ = edge_buckets_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(
                options->node_partition_ordering, graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                graph_storage_->getNumNodes(), options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
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

shared_ptr<Batch> DataLoader::getBatch(at::optional<torch::Device> device, bool perform_map) {
    shared_ptr<Batch> batch = getNextBatch();
    if (batch == nullptr) {
        return batch;
    }

    if (batch->task_ == LearningTask::LINK_PREDICTION) {
        edgeSample(batch);
    } else if (batch->task_ == LearningTask::NODE_CLASSIFICATION || batch->task_ == LearningTask::ENCODE) {
        nodeSample(batch);
    }

    loadCPUParameters(batch);

    if (device.has_value()) {
        if (device.value().is_cuda()) {
            batch->to(device.value());
            loadGPUParameters(batch);
            batch->dense_graph_.performMap();
        }
    }

    if (perform_map) {
        batch->dense_graph_.performMap();
    }

    return batch;
}

void DataLoader::edgeSample(shared_ptr<Batch> batch) {
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
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
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

void DataLoader::nodeSample(shared_ptr<Batch> batch) {
    if (batch->task_ == LearningTask::ENCODE) {
        torch::TensorOptions node_opts = torch::TensorOptions().dtype(torch::kInt64).device(graph_storage_->storage_ptrs_.edges->device_);
        batch->root_node_indices_ = torch::arange(batch->start_idx_, batch->start_idx_ + batch->batch_size_, node_opts);
    } else {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
    }

    if (graph_storage_->storage_ptrs_.node_labels != nullptr) {
        batch->node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_).flatten(0, 1);
    }

    if (graph_storage_->current_subgraph_state_->global_to_local_index_map_.defined()) {
        batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
    }

    if (neighbor_sampler_ != nullptr) {
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
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

void DataLoader::loadCPUParameters(shared_ptr<Batch> batch) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ != torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }

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
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
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

    if (!buffer_states_.empty()) {
        graph_storage_->initializeInMemorySubGraph(buffer_states_[0]);
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}));
    }

    if (negative_sampler_ != nullptr) {
        if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler_)) {
            if (std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler_)->filtered_) {
                graph_storage_->sortAllEdges();
            }
        }
    }
}