//
// Created by Jason Mohoney on 6/18/21.
//

#include "graph_storage.h"

#include <algorithm>
#include <random>

//#include <omp.h>

#include "ordering.h"
#include "logger.h"

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs,
                                     shared_ptr<StorageConfig> storage_config,
                                     LearningTask learning_task,
                                     bool filtered_eval) {
    storage_ptrs_ = storage_ptrs;
    storage_config_ = storage_config;
    learning_task_ = learning_task;
    filtered_eval_ = filtered_eval;
    train_ = true;
    num_edges_ = storage_config_->dataset->num_edges;
    num_nodes_ = storage_config_->dataset->num_nodes;
    full_graph_evaluation_ = storage_config_->full_graph_evaluation;

    prefetch_ = storage_config_->prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    if (filtered_eval_) {
        storage_ptrs_.train_edges->load();
        storage_ptrs_.validation_edges->load();
        storage_ptrs_.test_edges->load();
        EdgeList sorted_edges = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kCPU);
        if (storage_ptrs_.validation_edges->getDim0() > 0) {
            sorted_edges = torch::cat({sorted_edges, storage_ptrs_.validation_edges->range(0, storage_ptrs_.validation_edges->getDim0()).to(torch::kCPU)});
        }

        if (storage_ptrs_.test_edges->getDim0() > 0) {
            sorted_edges = torch::cat({sorted_edges, storage_ptrs_.test_edges->range(0, storage_ptrs_.test_edges->getDim0()).to(torch::kCPU)});
        }

        auto sorted_edges_accessor = sorted_edges.accessor<int, 2>();
        if (storage_ptrs_.train_edges->dim1_size_ == 3) {
            for (int64_t i = 0; i < sorted_edges.size(0); i++) {
                src_map_.emplace(std::make_pair(sorted_edges_accessor[i][0], sorted_edges_accessor[i][1]), vector<int>());
                dst_map_.emplace(std::make_pair(sorted_edges_accessor[i][2], sorted_edges_accessor[i][1]), vector<int>());
            }
            for (int64_t i = 0; i < sorted_edges.size(0); i++) {
                src_map_.at(std::make_pair(sorted_edges_accessor[i][0], sorted_edges_accessor[i][1])).emplace_back(sorted_edges_accessor[i][2]);
                dst_map_.at(std::make_pair(sorted_edges_accessor[i][2], sorted_edges_accessor[i][1])).emplace_back(sorted_edges_accessor[i][0]);
            }
        } else {
            for (int64_t i = 0; i < sorted_edges.size(0); i++) {
                src_map_.emplace(std::make_pair(sorted_edges_accessor[i][0], 0), vector<int>());
                dst_map_.emplace(std::make_pair(sorted_edges_accessor[i][1],0), vector<int>());
            }
            for (int64_t i = 0; i < sorted_edges.size(0); i++) {
                src_map_.at(std::make_pair(sorted_edges_accessor[i][0], 0)).emplace_back(sorted_edges_accessor[i][1]);
                dst_map_.at(std::make_pair(sorted_edges_accessor[i][1], 0)).emplace_back(sorted_edges_accessor[i][0]);
            }
        }

        storage_ptrs_.train_edges->unload(false);
        storage_ptrs_.validation_edges->unload(false);
        storage_ptrs_.test_edges->unload(false);
    }

    if (full_graph_evaluation_) {
        if (storage_config_->embeddings != nullptr) {
            if (storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
                string node_embedding_filename = storage_config_->dataset->base_directory
                                                 + PathConstants::nodes_directory
                                                 + PathConstants::embeddings_file
                                                 + PathConstants::file_ext;

                in_memory_embeddings_ = new InMemory(node_embedding_filename,
                                                     num_nodes_,
                                                     storage_ptrs_.node_embeddings->dim1_size_,
                                                     storage_config_->embeddings->options->dtype,
                                                     torch::kCPU);
            }
        }

        if (storage_config_->features != nullptr) {
            if (storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
                string node_feature_filename = storage_config_->dataset->base_directory
                                               + PathConstants::nodes_directory
                                               + PathConstants::features_file
                                               + PathConstants::file_ext;

                in_memory_features_ = new InMemory(node_feature_filename,
                                                   num_nodes_,
                                                   storage_config_->dataset->feature_dim,
                                                   storage_config_->features->options->dtype,
                                                   torch::kCPU);
            }
        }
    }
}

GraphModelStorage::~GraphModelStorage() {

    unload(false);

    delete storage_ptrs_.train_edges;
    delete storage_ptrs_.train_edges_dst_sort;
    delete storage_ptrs_.train_edges_features;
    delete storage_ptrs_.validation_edges;
    delete storage_ptrs_.validation_edges_features;
    delete storage_ptrs_.test_edges;
    delete storage_ptrs_.test_edges_features;
    delete storage_ptrs_.node_embeddings;
    delete storage_ptrs_.node_optimizer_state;
    delete storage_ptrs_.node_features;
    delete storage_ptrs_.relation_features;

    if (storage_config_->embeddings != nullptr) {
        if (full_graph_evaluation_ && storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            delete in_memory_embeddings_;
        }
    }

    if (storage_config_->features != nullptr) {
        if (full_graph_evaluation_ && storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
            delete in_memory_features_;
        }
    }

    if (current_subgraph_state_ != nullptr) {
        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            delete current_subgraph_state_->in_memory_subgraph_;
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }
        delete current_subgraph_state_;
        current_subgraph_state_ = nullptr;
    }

    if (next_subgraph_state_ != nullptr) {
        if (next_subgraph_state_->in_memory_subgraph_ != nullptr) {
            delete next_subgraph_state_->in_memory_subgraph_;
            next_subgraph_state_->in_memory_subgraph_ = nullptr;
        }
        delete next_subgraph_state_;
        next_subgraph_state_ = nullptr;
    }

    delete subgraph_lock_;
    delete subgraph_cv_;
}

void GraphModelStorage::_load(Storage *storage) {
    if (storage != nullptr) {
        storage->load();
    }
}

void GraphModelStorage::_unload(Storage *storage, bool write) {
    if (storage != nullptr) {
        storage->unload(write);
    }
}

void GraphModelStorage::load() {
    _load(storage_ptrs_.edges);
    _load(storage_ptrs_.train_edges);
    _load(storage_ptrs_.train_edges_dst_sort);
    _load(storage_ptrs_.edge_features);
    _load(storage_ptrs_.nodes);

    if (train_) {
        _load(storage_ptrs_.node_embeddings);
        _load(storage_ptrs_.node_optimizer_state);
        _load(storage_ptrs_.node_features);
    } else {
        if (storage_config_->embeddings != nullptr) {
            if (storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER && full_graph_evaluation_) {
                _load(in_memory_embeddings_);
            } else {
                _load(storage_ptrs_.node_embeddings);
            }
        }

        if (storage_config_->features != nullptr) {
            if (storage_config_->features->type == StorageBackend::PARTITION_BUFFER && full_graph_evaluation_) {
                _load(in_memory_features_);
            } else {
                _load(storage_ptrs_.node_features);
            }
        }
    }

    _load(storage_ptrs_.node_labels);
    _load(storage_ptrs_.relation_features);
}

void GraphModelStorage::unload(bool write) {
    // only unload features/embeddings etc if using the partition buffer
    bool should_unload = false;
    if (storage_config_->embeddings != nullptr) {
        if (storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            should_unload = true;
        }
    }
    if (storage_config_->features != nullptr) {
        if (storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
            should_unload = true;
        }
    }

    if (should_unload) {
        _unload(storage_ptrs_.edges, false);
        _unload(storage_ptrs_.train_edges, false);
        _unload(storage_ptrs_.train_edges_dst_sort, false);
        _unload(storage_ptrs_.validation_edges, false);
        _unload(storage_ptrs_.test_edges, false);
        _unload(storage_ptrs_.nodes, false);
        _unload(storage_ptrs_.train_nodes, false);
        _unload(storage_ptrs_.valid_nodes, false);
        _unload(storage_ptrs_.test_nodes, false);
        _unload(storage_ptrs_.node_embeddings, write);
        _unload(storage_ptrs_.node_optimizer_state, write);
        _unload(storage_ptrs_.node_features, false);
    }

    _unload(storage_ptrs_.edge_features, false);
    _unload(storage_ptrs_.train_edges_features, false);
    _unload(storage_ptrs_.validation_edges_features, false);
    _unload(storage_ptrs_.test_edges_features, false);
    _unload(storage_ptrs_.relation_features, false);

    _unload(in_memory_embeddings_, false);
    _unload(in_memory_features_, false);

    active_edges_ = torch::Tensor();
    active_nodes_ = torch::Tensor();

    if (current_subgraph_state_ != nullptr) {
        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            delete current_subgraph_state_->in_memory_subgraph_;
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }
        delete current_subgraph_state_;
        current_subgraph_state_ = nullptr;
    }

    if (next_subgraph_state_ != nullptr) {
        if (next_subgraph_state_->in_memory_subgraph_ != nullptr) {
            delete next_subgraph_state_->in_memory_subgraph_;
            next_subgraph_state_->in_memory_subgraph_ = nullptr;
        }
        delete next_subgraph_state_;
        next_subgraph_state_ = nullptr;
    }
}

void GraphModelStorage::setEvalFilter(Batch *batch) {
    auto temp_src = batch->src_pos_indices_.to(torch::kCPU);
    auto temp_dst = batch->dst_pos_indices_.to(torch::kCPU);
    auto src_pos_ind_accessor = temp_src.accessor<int64_t, 1>();
    auto dst_pos_ind_accessor = temp_dst.accessor<int64_t, 1>();

    batch->src_neg_filter_eval_ = std::vector<torch::Tensor>(batch->batch_size_);
    batch->dst_neg_filter_eval_ = std::vector<torch::Tensor>(batch->batch_size_);

    if (batch->rel_indices_.defined()) {
        auto temp_rel = batch->rel_indices_.to(torch::kCPU);
        auto rel_ind_accessor = temp_rel.accessor<int64_t, 1>();

        #pragma omp parallel for
        for (int64_t i = 0; i < batch->batch_size_; i++) {
            auto src_map_vec = src_map_.at(std::make_pair(src_pos_ind_accessor[i], rel_ind_accessor[i]));
            auto dst_map_vec = dst_map_.at(std::make_pair(dst_pos_ind_accessor[i], rel_ind_accessor[i]));
            auto src_map_tens = torch::tensor(src_map_vec);
            auto dst_map_tens = torch::tensor(dst_map_vec);
            batch->src_neg_filter_eval_[i] = dst_map_tens;
            batch->dst_neg_filter_eval_[i] = src_map_tens;
        }
    } else {
        auto temp_rel = batch->rel_indices_.to(torch::kCPU);
        auto rel_ind_accessor = temp_rel.accessor<int64_t, 1>();

        #pragma omp parallel for
        for (int64_t i = 0; i < batch->batch_size_; i++) {
            auto src_map_vec = src_map_.at(std::make_pair(src_pos_ind_accessor[i], 0));
            auto dst_map_vec = dst_map_.at(std::make_pair(dst_pos_ind_accessor[i], 0));
            auto src_map_tens = torch::tensor(src_map_vec);
            auto dst_map_tens = torch::tensor(dst_map_vec);
            batch->src_neg_filter_eval_[i] = dst_map_tens;
            batch->dst_neg_filter_eval_[i] = src_map_tens;
        }
    }

}

void GraphModelStorage::setEdgesStorage(Storage *edge_storage) {
    storage_ptrs_.edges = edge_storage;
    num_edges_ = storage_ptrs_.edges->getDim0();
}

void GraphModelStorage::setNodesStorage(Storage *node_storage) {
    storage_ptrs_.nodes = node_storage;
}

EdgeList GraphModelStorage::getEdges(Indices indices) {
    if (active_edges_.defined()) {
        return active_edges_.index_select(0, indices);
    } else {
        return storage_ptrs_.edges->indexRead(indices);
    }
}

EdgeList GraphModelStorage::getEdgesRange(int64_t start, int64_t size) {
    if (active_edges_.defined()) {
        return active_edges_.narrow(0, start, size);
    } else {
        return storage_ptrs_.edges->range(start, size);
    }
}

void GraphModelStorage::shuffleEdges() {
    storage_ptrs_.edges->shuffle();
}

Indices GraphModelStorage::getRandomNodeIds(int64_t size) {
    torch::TensorOptions ind_opts;
    if (storage_config_->edges->type == StorageBackend::DEVICE_MEMORY) {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(storage_config_->device_type);
    } else {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    }

    Indices ret;
    if (useInMemorySubGraph()) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            ret = ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getRandomIds(size);
        } else {
            ret = ((PartitionBufferStorage *) storage_ptrs_.node_features)->getRandomIds(size);
        }
    } else {
        ret = torch::randint(num_nodes_, {size}, ind_opts);
    }

    return ret;
}

Indices GraphModelStorage::getNodeIdsRange(int64_t start, int64_t size) {
    if (active_nodes_.defined()) {
        return active_nodes_.narrow(0, start, size);
    } else {
        return storage_ptrs_.nodes->range(start, size).flatten(0, 1);
    }
}

Embeddings GraphModelStorage::getNodeEmbeddings(Indices indices) {
    if (!train_ && storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER && full_graph_evaluation_) {
        if (in_memory_embeddings_ != nullptr) {
            return in_memory_embeddings_->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

Embeddings GraphModelStorage::getNodeEmbeddingsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->range(start, size);
    } else {
        return torch::Tensor();
    }
}

Features GraphModelStorage::getNodeFeatures(Indices indices) {
    if (!train_ && storage_config_->features->type == StorageBackend::PARTITION_BUFFER && full_graph_evaluation_) {
        if (in_memory_features_ != nullptr) {
            return in_memory_features_->indexRead(indices);

        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

Features GraphModelStorage::getNodeFeaturesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->range(start, size);
    } else {
        return torch::Tensor();
    }
}

Labels GraphModelStorage::getNodeLabels(Indices indices) {

    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

Labels GraphModelStorage::getNodeLabelsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddings(Indices indices, Embeddings embeddings) {
    storage_ptrs_.node_embeddings->indexPut(indices, embeddings);
}

void GraphModelStorage::updateAddNodeEmbeddings(Indices indices, torch::Tensor values) {
    storage_ptrs_.node_embeddings->indexAdd(indices, values);
}

OptimizerState GraphModelStorage::getNodeEmbeddingState(Indices indices) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddingState(Indices indices, OptimizerState state) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexPut(indices, state);
    }
}

void GraphModelStorage::updateAddNodeEmbeddingState(Indices indices, torch::Tensor values) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
    }
}

bool GraphModelStorage::embeddingsOffDevice() {

    if (storage_config_->embeddings != nullptr) {
        return ((storage_config_->embeddings->type != StorageBackend::DEVICE_MEMORY) && (storage_config_->device_type == torch::kCUDA));
    } else if (storage_config_->features != nullptr) {
        return ((storage_config_->features->type != StorageBackend::DEVICE_MEMORY) && (storage_config_->device_type == torch::kCUDA));
    } else {
        return false;
    }
}

void GraphModelStorage::initializeInMemorySubGraph(torch::Tensor buffer_state) {

    if (useInMemorySubGraph()) {
        current_subgraph_state_ = new InMemorySubgraphState();

        buffer_state = buffer_state.to(torch::kInt64);

        int buffer_size = buffer_state.size(0);
        int num_edge_buckets_in_mem = buffer_size * buffer_size;
        int num_partitions = getNumPartitions();

        torch::Tensor new_in_mem_partition_ids = buffer_state;
        auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

        torch::Tensor in_mem_edge_bucket_ids = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor in_mem_edge_bucket_sizes = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor global_edge_bucket_starts = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);

//        torch::Tensor in_mem_edge_bucket_ids_dst = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
//        torch::Tensor in_mem_edge_bucket_sizes_dst = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
//        torch::Tensor global_edge_bucket_starts_dst = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);

        auto in_mem_edge_bucket_ids_accessor = in_mem_edge_bucket_ids.accessor<int64_t, 1>();
        auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
        auto global_edge_bucket_starts_accessor = global_edge_bucket_starts.accessor<int64_t, 1>();

//        auto in_mem_edge_bucket_ids_dst_accessor = in_mem_edge_bucket_ids_dst.accessor<int64_t, 1>();
//        auto in_mem_edge_bucket_sizes_dst_accessor = in_mem_edge_bucket_sizes_dst.accessor<int64_t, 1>();
//        auto global_edge_bucket_starts_dst_accessor = global_edge_bucket_starts_dst.accessor<int64_t, 1>();

        //TODO we don't need to do this every time
        std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
        torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int) edge_bucket_sizes_.size()}, torch::kInt64);
        torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
        torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

        #pragma omp parallel for
        for (int i = 0; i < buffer_size; i++) {
            for (int j = 0; j < buffer_size; j++) {
                int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];
                int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
                int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

                int idx = i * buffer_size + j;
                in_mem_edge_bucket_ids_accessor[idx] = edge_bucket_id;
                in_mem_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
                global_edge_bucket_starts_accessor[idx] = edge_bucket_start;

//                idx = j * buffer_size + i;
//                in_mem_edge_bucket_ids_dst_accessor[idx] = edge_bucket_id;
//                in_mem_edge_bucket_sizes_dst_accessor[idx] = edge_bucket_size;
//                global_edge_bucket_starts_dst_accessor[idx] = edge_bucket_start;
            }
        }

        torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
        int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
        in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

//        torch::Tensor in_mem_edge_bucket_starts_dst = in_mem_edge_bucket_sizes_dst.cumsum(0);
//        in_mem_edge_bucket_starts_dst = in_mem_edge_bucket_starts_dst - in_mem_edge_bucket_sizes_dst;

        auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();
//        auto in_mem_edge_bucket_starts_dst_accessor = in_mem_edge_bucket_starts_dst.accessor<int64_t, 1>();

        current_subgraph_state_->all_in_memory_edges_ = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);
//        current_subgraph_state_->all_in_memory_edges_dst_sort_ = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

        #pragma omp parallel for
        for (int i = 0; i < num_edge_buckets_in_mem; i++) {
            int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = global_edge_bucket_starts_accessor[i];
            int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];

            current_subgraph_state_->all_in_memory_edges_.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.train_edges->range(edge_bucket_start, edge_bucket_size);

//            edge_bucket_size = in_mem_edge_bucket_sizes_dst_accessor[i];
//            edge_bucket_start = global_edge_bucket_starts_dst_accessor[i];
//            local_offset = in_mem_edge_bucket_starts_dst_accessor[i];
//            current_subgraph_state_->all_in_memory_edges_dst_sort_.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.train_edges_dst_sort->range(edge_bucket_start, edge_bucket_size);
        }

        if (storage_ptrs_.node_embeddings != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ = ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getGlobalToLocalMap(true);
        } else if (storage_ptrs_.node_features != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ = ((PartitionBufferStorage *) storage_ptrs_.node_features)->getGlobalToLocalMap(true);
        }

        torch::Tensor mapped_edges;
        torch::Tensor mapped_edges_dst_sort;
        if (storage_ptrs_.edges->dim1_size_ == 3) {
            mapped_edges = torch::stack({
                  current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                  current_subgraph_state_->all_in_memory_edges_.select(1, 1),
                  current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))
            }).transpose(0, 1);

//            mapped_edges_dst_sort = torch::stack({
//                   current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_dst_sort_.select(1, 0)),
//                   current_subgraph_state_->all_in_memory_edges_dst_sort_.select(1, 1),
//                   current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_dst_sort_.select(1, -1))
//           }).transpose(0, 1);
        } else if (storage_ptrs_.edges->dim1_size_ == 2) {
            mapped_edges = torch::stack({
                  current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                  current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))
            }).transpose(0, 1);

//            mapped_edges_dst_sort = torch::stack({
//                   current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_dst_sort_.select(1, 0)),
//                   current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_dst_sort_.select(1, -1))
//            }).transpose(0, 1);
        } else {
            // TODO use a function for logging errors and throwing expections
            SPDLOG_ERROR("Unexpected number of edge columns");
            std::runtime_error("Unexpected number of edge columns");
        }


//        assert((mapped_edges == -1).nonzero().size(0) == 0);
//        assert((mapped_edges_dst_sort == -1).nonzero().size(0) == 0);

        current_subgraph_state_->all_in_memory_mapped_edges_ = mapped_edges;

        mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
//        mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges_dst_sort, in_mem_edge_bucket_starts_dst, buffer_size, false);
        mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);

        mapped_edges = mapped_edges.to(torch::kInt64);
        mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            delete current_subgraph_state_->in_memory_subgraph_;
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }
        current_subgraph_state_->in_memory_subgraph_ = new MariusGraph(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory());

        current_subgraph_state_->in_memory_partition_ids_ = new_in_mem_partition_ids;
        current_subgraph_state_->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
        current_subgraph_state_->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
        current_subgraph_state_->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;
//        current_subgraph_state_->in_memory_edge_bucket_ids_dst_ = in_mem_edge_bucket_ids_dst;
//        current_subgraph_state_->in_memory_edge_bucket_sizes_dst_ = in_mem_edge_bucket_sizes_dst;
//        current_subgraph_state_->in_memory_edge_bucket_starts_dst_ = in_mem_edge_bucket_starts_dst;

        if (prefetch_) {
            if (hasSwap()) {
                // update next_subgraph_state_ in background
                getNextSubGraph();
            }
        }
    } else {
        current_subgraph_state_ = new InMemorySubgraphState();

        bool embeddings_buffered = false;
        bool features_buffered = false;

        if (storage_config_->embeddings != nullptr) {
            StorageBackend embeddings_backend = storage_config_->embeddings->type;
            embeddings_buffered = (embeddings_backend == StorageBackend::PARTITION_BUFFER);
        }

        if (storage_config_->features != nullptr) {
            StorageBackend features_backend = storage_config_->features->type;
            features_buffered = (features_backend == StorageBackend::PARTITION_BUFFER);
        }

        EdgeList src_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0());
        EdgeList dst_sort;
//        EdgeList dst_sort = storage_ptrs_.train_edges_dst_sort->range(0, storage_ptrs_.train_edges_dst_sort->getDim0()).to(torch::kInt64);

//        src_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, 0)));
//        EdgeList dst_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, -1)));

        if (!embeddings_buffered && !features_buffered) {
            dst_sort = storage_ptrs_.train_edges_dst_sort->range(0, storage_ptrs_.train_edges_dst_sort->getDim0());

        } else if (!train_ && full_graph_evaluation_) {
            // TODO: need to merge to sorted edge buckets
            src_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, 0)));
            dst_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, -1)));
        } else {
            SPDLOG_ERROR("Unsure how to create in memory subgraph.");
            throw std::runtime_error("");
        }

        src_sort = src_sort.to(torch::kInt64);
        dst_sort = dst_sort.to(torch::kInt64);

        current_subgraph_state_->in_memory_subgraph_ = new MariusGraph(src_sort, dst_sort, getNumNodesInMemory());
    }
}

void GraphModelStorage::updateInMemorySubGraph() {


    if (prefetch_) {
        // wait until the prefetching has been completed
        std::unique_lock lock(*subgraph_lock_);
        subgraph_cv_->wait(lock, [this] { return prefetch_complete_ == true;});
        // need to wait for the subgraph to be prefetched to perform the swap, otherwise the prefetched buffer_index_map may be incorrect
        performSwap();
        // free previous subgraph
        delete current_subgraph_state_->in_memory_subgraph_;
        delete current_subgraph_state_;

        current_subgraph_state_ = next_subgraph_state_;
        next_subgraph_state_ = nullptr;
        prefetch_complete_ = false;

        if (hasSwap()) {
            // update next_subgraph_state_ in background
            getNextSubGraph();
        }
    } else {
        std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds();
        performSwap();
        updateInMemorySubGraph_(current_subgraph_state_, current_swap_ids);
    }
}

void GraphModelStorage::getNextSubGraph() {
    std::pair<std::vector<int>, std::vector<int>> next_swap_ids = getNextSwapIds();
    next_subgraph_state_ = new InMemorySubgraphState();
    next_subgraph_state_->in_memory_subgraph_ = nullptr;
    std::thread(&GraphModelStorage::updateInMemorySubGraph_, this, next_subgraph_state_, next_swap_ids).detach();
}

void GraphModelStorage::updateInMemorySubGraph_(InMemorySubgraphState *subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids) {

    if (prefetch_) {
        subgraph_lock_->lock();
    }

    std::vector<int> evict_partition_ids = std::get<0>(swap_ids);
    std::vector<int> admit_partition_ids = std::get<1>(swap_ids);

    torch::Tensor admit_ids_tensor = torch::tensor(admit_partition_ids, torch::kCPU);

    int buffer_size = current_subgraph_state_->in_memory_partition_ids_.size(0);
    int num_edge_buckets_in_mem = current_subgraph_state_->in_memory_edge_bucket_ids_.size(0);
    int num_partitions = getNumPartitions();
    int num_swap_partitions = evict_partition_ids.size();
    int num_remaining_partitions = buffer_size - num_swap_partitions;

    // get edge buckets that will be kept in memory
    torch::Tensor keep_mask = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
    auto accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_edge_bucket_ids_ = current_subgraph_state_->in_memory_edge_bucket_ids_.accessor<int64_t, 1>();

//    torch::Tensor keep_mask_dst = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
//    auto accessor_keep_mask_dst = keep_mask_dst.accessor<bool, 1>();
//    auto accessor_in_memory_edge_bucket_ids_dst_ = current_subgraph_state_->in_memory_edge_bucket_ids_dst_.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_id = accessor_in_memory_edge_bucket_ids_[i];
        int64_t src_partition = edge_bucket_id / num_partitions;
        int64_t dst_partition = edge_bucket_id % num_partitions;

//        edge_bucket_id = accessor_in_memory_edge_bucket_ids_dst_[i];
//        int64_t src_partition_dst = edge_bucket_id / num_partitions;
//        int64_t dst_partition_dst = edge_bucket_id % num_partitions;

        for (int j = 0; j < num_swap_partitions; j++) {
            if (src_partition == evict_partition_ids[j] || dst_partition == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
            }
//            if (src_partition_dst == evict_partition_ids[j] || dst_partition_dst == evict_partition_ids[j]) {
//                accessor_keep_mask_dst[i] = false;
//            }
        }
    }

    torch::Tensor in_mem_edge_bucket_ids = current_subgraph_state_->in_memory_edge_bucket_ids_.masked_select(keep_mask);
    torch::Tensor in_mem_edge_bucket_sizes = current_subgraph_state_->in_memory_edge_bucket_sizes_.masked_select(keep_mask);
    torch::Tensor local_or_global_edge_bucket_starts = current_subgraph_state_->in_memory_edge_bucket_starts_.masked_select(keep_mask);

//    torch::Tensor in_mem_edge_bucket_ids_dst = current_subgraph_state_->in_memory_edge_bucket_ids_dst_.masked_select(keep_mask_dst);
//    torch::Tensor in_mem_edge_bucket_sizes_dst = current_subgraph_state_->in_memory_edge_bucket_sizes_dst_.masked_select(keep_mask_dst);
//    torch::Tensor local_or_global_edge_bucket_starts_dst = current_subgraph_state_->in_memory_edge_bucket_starts_dst_.masked_select(keep_mask_dst);

    // get new in memory partition ids
    keep_mask = torch::ones({buffer_size}, torch::kBool);
    accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_partition_ids_ = current_subgraph_state_->in_memory_partition_ids_.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        int64_t partition_id = accessor_in_memory_partition_ids_[i];

        for (int j = 0; j < num_swap_partitions; j++) {
            if (partition_id == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
                break;
            }
        }
    }

    torch::Tensor old_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_select(keep_mask);
    torch::Tensor new_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_scatter(~keep_mask, admit_ids_tensor);
    auto old_in_mem_partition_ids_accessor = old_in_mem_partition_ids.accessor<int64_t, 1>();
    auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

    // get new incoming edge buckets
    int num_new_edge_buckets = num_swap_partitions * (num_remaining_partitions + buffer_size);

    torch::Tensor new_edge_bucket_ids = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_edge_bucket_sizes = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_global_edge_bucket_starts = torch::zeros({num_new_edge_buckets}, torch::kInt64);

    auto new_edge_bucket_ids_accessor = new_edge_bucket_ids.accessor<int64_t, 1>();
    auto new_edge_bucket_sizes_accessor = new_edge_bucket_sizes.accessor<int64_t, 1>();
    auto new_global_edge_bucket_starts_accessor = new_global_edge_bucket_starts.accessor<int64_t, 1>();

    //TODO we don't need to do this every time
    std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
    torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int) edge_bucket_sizes_.size()}, torch::kInt64);
    torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
    torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
    auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
    auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int i = 0; i < num_remaining_partitions; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = old_in_mem_partition_ids_accessor[i] * num_partitions + admit_partition_ids[j];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    int offset = num_swap_partitions * num_remaining_partitions;

    #pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = admit_partition_ids[j] * num_partitions + new_in_mem_partition_ids_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = offset + i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    // concatenate old and new
    in_mem_edge_bucket_ids = torch::cat({in_mem_edge_bucket_ids, new_edge_bucket_ids});
    in_mem_edge_bucket_sizes = torch::cat({in_mem_edge_bucket_sizes, new_edge_bucket_sizes});
    local_or_global_edge_bucket_starts = torch::cat({local_or_global_edge_bucket_starts, new_global_edge_bucket_starts});

//    in_mem_edge_bucket_ids_dst = torch::cat({in_mem_edge_bucket_ids_dst, new_edge_bucket_ids});
//    in_mem_edge_bucket_sizes_dst = torch::cat({in_mem_edge_bucket_sizes_dst, new_edge_bucket_sizes});
//    local_or_global_edge_bucket_starts_dst = torch::cat({local_or_global_edge_bucket_starts_dst, new_global_edge_bucket_starts});

    torch::Tensor in_mem_mask = torch::ones({num_edge_buckets_in_mem - num_new_edge_buckets}, torch::kBool);
    in_mem_mask = torch::cat({in_mem_mask, torch::zeros({num_new_edge_buckets}, torch::kBool)});
//    torch::Tensor in_mem_mask_dst = in_mem_mask.clone();

    // put the ids in the correct order so the mapped edges remain sorted
    torch::Tensor src_ids_order = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
//    torch::Tensor dst_ids_order = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
    auto src_ids_order_accessor = src_ids_order.accessor<int64_t, 1>();
//    auto dst_ids_order_accessor = dst_ids_order.accessor<int64_t, 1>();

    #pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < buffer_size; j++) {
            int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];

            int idx = i * buffer_size + j;
            src_ids_order_accessor[idx] = edge_bucket_id;

//            idx = j * buffer_size + i;
//            dst_ids_order_accessor[idx] = edge_bucket_id;
        }
    }

    // TODO: all these argsorts can be done with one omp for loop, probably faster, same with masked_selects above
    torch::Tensor arg_sort = torch::argsort(in_mem_edge_bucket_ids);
    arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(src_ids_order))));
    in_mem_edge_bucket_ids = (in_mem_edge_bucket_ids.index_select(0, arg_sort));
    in_mem_edge_bucket_sizes = (in_mem_edge_bucket_sizes.index_select(0, arg_sort));
    local_or_global_edge_bucket_starts = (local_or_global_edge_bucket_starts.index_select(0, arg_sort));
    in_mem_mask = (in_mem_mask.index_select(0, arg_sort));

//    arg_sort = torch::argsort(in_mem_edge_bucket_ids_dst);
//    arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(dst_ids_order))));
//    in_mem_edge_bucket_ids_dst = (in_mem_edge_bucket_ids_dst.index_select(0, arg_sort));
//    in_mem_edge_bucket_sizes_dst = (in_mem_edge_bucket_sizes_dst.index_select(0, arg_sort));
//    local_or_global_edge_bucket_starts_dst = (local_or_global_edge_bucket_starts_dst.index_select(0, arg_sort));
//    in_mem_mask_dst = (in_mem_mask_dst.index_select(0, arg_sort));

    // with everything in order grab the edge buckets
    torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
    int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
    in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

//    torch::Tensor in_mem_edge_bucket_starts_dst = in_mem_edge_bucket_sizes_dst.cumsum(0);
//    in_mem_edge_bucket_starts_dst = in_mem_edge_bucket_starts_dst - in_mem_edge_bucket_sizes_dst;

    auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
    auto local_or_global_edge_bucket_starts_accessor = local_or_global_edge_bucket_starts.accessor<int64_t, 1>();
    auto in_mem_mask_accessor = in_mem_mask.accessor<bool, 1>();
    auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

//    auto in_mem_edge_bucket_sizes_dst_accessor = in_mem_edge_bucket_sizes_dst.accessor<int64_t, 1>();
//    auto local_or_global_edge_bucket_starts_dst_accessor = local_or_global_edge_bucket_starts_dst.accessor<int64_t, 1>();
//    auto in_mem_mask_dst_accessor = in_mem_mask_dst.accessor<bool, 1>();
//    auto in_mem_edge_bucket_starts_dst_accessor = in_mem_edge_bucket_starts_dst.accessor<int64_t, 1>();

    torch::Tensor new_all_in_memory_edges = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);
//    torch::Tensor new_all_in_memory_edges_dst_sort = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

    // get the edges
    #pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
        int64_t edge_bucket_start = local_or_global_edge_bucket_starts_accessor[i];
        bool in_mem = in_mem_mask_accessor[i];
        int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];

        if (in_mem) {
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) = current_subgraph_state_->all_in_memory_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        } else {
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.train_edges->range(edge_bucket_start, edge_bucket_size);
        }

//        edge_bucket_size = in_mem_edge_bucket_sizes_dst_accessor[i];
//        edge_bucket_start = local_or_global_edge_bucket_starts_dst_accessor[i];
//        in_mem = in_mem_mask_dst_accessor[i];
//        local_offset = in_mem_edge_bucket_starts_dst_accessor[i];

//        if (in_mem) {
//            new_all_in_memory_edges_dst_sort.narrow(0, local_offset, edge_bucket_size) = current_subgraph_state_->all_in_memory_edges_dst_sort_.narrow(0, edge_bucket_start, edge_bucket_size);
//        } else {
//            new_all_in_memory_edges_dst_sort.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.train_edges_dst_sort->range(edge_bucket_start, edge_bucket_size);
//        }
    }

    subgraph->all_in_memory_edges_ = new_all_in_memory_edges;
//    subgraph->all_in_memory_edges_dst_sort_ = new_all_in_memory_edges_dst_sort;

    if (storage_ptrs_.node_embeddings != nullptr) {
        subgraph->global_to_local_index_map_ = ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getGlobalToLocalMap(!prefetch_);
    } else if (storage_ptrs_.node_features != nullptr) {
        subgraph->global_to_local_index_map_ = ((PartitionBufferStorage *) storage_ptrs_.node_features)->getGlobalToLocalMap(!prefetch_);
    }


    torch::Tensor mapped_edges;
    torch::Tensor mapped_edges_dst_sort;
    if (storage_ptrs_.edges->dim1_size_ == 3) {
        mapped_edges = torch::stack({
            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
            subgraph->all_in_memory_edges_.select(1, 1),
            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))
        }).transpose(0, 1);

//        mapped_edges_dst_sort = torch::stack({
//            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_dst_sort_.select(1, 0)),
//            subgraph->all_in_memory_edges_dst_sort_.select(1, 1),
//            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_dst_sort_.select(1, -1))
//         }).transpose(0, 1);
    } else if (storage_ptrs_.edges->dim1_size_ == 2) {
        mapped_edges = torch::stack({
            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))
        }).transpose(0, 1);

//        mapped_edges_dst_sort = torch::stack({
//            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_dst_sort_.select(1, 0)),
//            subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_dst_sort_.select(1, -1))
//         }).transpose(0, 1);
    } else {
        // TODO use a function for logging errors and throwing expections
        SPDLOG_ERROR("Unexpected number of edge columns");
        std::runtime_error("Unexpected number of edge columns");
    }

//    assert((mapped_edges == -1).nonzero().size(0) == 0);
//    assert((mapped_edges_dst_sort == -1).nonzero().size(0) == 0);

    subgraph->all_in_memory_mapped_edges_ = mapped_edges;

    mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
//    mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges_dst_sort, in_mem_edge_bucket_starts_dst, buffer_size, false);
    mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);

    mapped_edges = mapped_edges.to(torch::kInt64);
    mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

    if (subgraph->in_memory_subgraph_ != nullptr) {
        delete subgraph->in_memory_subgraph_;
        subgraph->in_memory_subgraph_ = nullptr;
    }

    subgraph->in_memory_subgraph_ = new MariusGraph(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory());

    // update state
    subgraph->in_memory_partition_ids_ = new_in_mem_partition_ids;
    subgraph->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
    subgraph->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
    subgraph->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;
//    subgraph->in_memory_edge_bucket_ids_dst_ = in_mem_edge_bucket_ids_dst;
//    subgraph->in_memory_edge_bucket_sizes_dst_ = in_mem_edge_bucket_sizes_dst;
//    subgraph->in_memory_edge_bucket_starts_dst_ = in_mem_edge_bucket_starts_dst;

    if (prefetch_) {
        prefetch_complete_ = true;
        subgraph_lock_->unlock();
        subgraph_cv_->notify_all();
    }
}

EdgeList GraphModelStorage::merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src) {
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }
    return edges.index_select(0, torch::argsort(edges.select(1, sort_dim)));

    /*EdgeList sorted_edges = torch::empty_like(edges);

//    int64_t *sorted_edges_mem = sorted_edges.data_ptr<int64_t>();
//    int64_t *edges_mem = edges.data_ptr<int64_t>();
    auto sorted_edges_accessor = sorted_edges.accessor<int64_t, 2>();
    auto edges_accessor = edges.accessor<int64_t, 2>();
    auto starts_accessor = starts.accessor<int64_t, 1>();

    int64_t max_node_val = num_nodes_;

    #pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        int chunk_start_idx = i * buffer_size;
        int64_t local_offset = starts_accessor[chunk_start_idx];

        vector<int64_t> sub_chunk_indices(buffer_size, 0);
        vector<int64_t> sub_chunk_limits(buffer_size, 0);
        for (int j = 0; j < buffer_size; j++) {
            sub_chunk_indices[j] = starts_accessor[chunk_start_idx + j];

            if (chunk_start_idx + j + 1 < starts.size(0)) {
                sub_chunk_limits[j] = starts_accessor[chunk_start_idx + j + 1];
            } else {
                sub_chunk_limits[j] = edges.size(0);
            }
        }

        int64_t count = 0;

        while (true) {
            int min_val_sub_chunk = -1;
            int64_t min_val = max_node_val;

            for (int j = 0; j < buffer_size; j++) {
                int64_t sub_chunk_idx = sub_chunk_indices[j];

                if (sub_chunk_idx < sub_chunk_limits[j]) {
//                    int64_t sub_chunk_val = *(edges_mem + (3 * sub_chunk_idx) + sort_dim);
                    int64_t sub_chunk_val = edges_accessor[sub_chunk_idx][sort_dim];

                    if (sub_chunk_val < min_val) {
                        min_val_sub_chunk = j;
                        min_val = sub_chunk_val;
                    }
                }
            }

            if (min_val_sub_chunk == -1) {
                break;
            }

            int64_t next_edge_idx = sub_chunk_indices[min_val_sub_chunk];

//            *(sorted_edges_mem + (3 * (local_offset + count))) = *(edges_mem + (3 * next_edge_idx));
//            *(sorted_edges_mem + (3 * (local_offset + count)) + 1) = *(edges_mem + (3 * next_edge_idx) + 1);
//            *(sorted_edges_mem + (3 * (local_offset + count)) + 2) = *(edges_mem + (3 * next_edge_idx) + 2);
            sorted_edges_accessor[local_offset + count][0] = edges_accessor[next_edge_idx][0];
            sorted_edges_accessor[local_offset + count][1] = edges_accessor[next_edge_idx][1];
            sorted_edges_accessor[local_offset + count][2] = edges_accessor[next_edge_idx][2];
//            edges_accessor[local_offset + count][0] = edges_accessor[next_edge_idx][0];
//            edges_accessor[local_offset + count][1] = edges_accessor[next_edge_idx][1];
//            edges_accessor[local_offset + count][2] = edges_accessor[next_edge_idx][2];

            count++;
            sub_chunk_indices[min_val_sub_chunk] = next_edge_idx + 1;
        }

    }

//    return sorted_edges;
    return sorted_edges.index_select(0, torch::arange(sorted_edges.size(0)));*/
}