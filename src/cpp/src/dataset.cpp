//
// Created by Jason Mohoney on 2/7/20.
//
#include <dataset.h>

using std::get;
using std::make_pair;
using std::forward_as_tuple;
using std::tie;

// training constructor
DataSet::DataSet(Storage *edges, Storage *embeddings, Storage *emb_state, Storage *src_relations, Storage *src_rel_state, Storage *dst_relations, Storage *dst_rel_state) {
    train_ = true;
    current_edge_ = 0;
    current_negative_id_ = 0;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    batch_lock_ = new std::mutex();
    negative_lock_ = new std::mutex();
    storage_loaded_ = false;

    edges_ = edges;
    num_edges_ = edges_->getDim0();
    node_embeddings_ = embeddings;
    num_nodes_ = embeddings->getDim0();
    node_embeddings_optimizer_state_ = emb_state;
    src_relations_ = src_relations;
    src_relations_optimizer_state_ = src_rel_state;
    dst_relations_ = dst_relations;
    dst_relations_optimizer_state_ = dst_rel_state;
    num_relations_ = src_relations_->getDim0();

    initializeBatches();
    batch_iterator_ = batches_.begin();
    timestamp_ = global_timestamp_allocator.getTimestamp();

    if (marius_options.storage.embeddings == BackendType::PartitionBuffer) {
        batches_ = ((PartitionBufferStorage *) node_embeddings_)->shuffleBeforeEvictions(batches_);
        batch_iterator_ = batches_.begin();
        ((PartitionBufferStorage *) node_embeddings_)->setOrdering(batches_);
        ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->setOrdering(batches_);
    }
}

// evaluation constructor
DataSet::DataSet(Storage *train_edges, Storage *eval_edges, Storage *test_edges, Storage *embeddings, Storage *src_relations, Storage *dst_relations) {
    train_ = false;
    current_edge_ = 0;
    current_negative_id_ = 0;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    batch_lock_ = new std::mutex();
    negative_lock_ = new std::mutex();
    storage_loaded_ = false;

    train_edges_ = train_edges;
    validation_edges_ = eval_edges;
    test_edges_ = test_edges;
    edges_ = validation_edges_;
    num_edges_ = edges_->getDim0();
    SPDLOG_DEBUG("Loaded Edges");

    if (marius_options.evaluation.filtered_evaluation) {
        train_edges_->load();
        validation_edges_->load();
        test_edges_->load();
        EdgeList sorted_edges = train_edges_->range(0, train_edges_->getDim0()).to(torch::kCPU);
        if (eval_edges->getDim0() > 0) {
            sorted_edges = torch::cat({sorted_edges, validation_edges_->range(0, validation_edges_->getDim0()).to(torch::kCPU)});
        }

        if (test_edges->getDim0() > 0) {
            sorted_edges = torch::cat({sorted_edges, test_edges->range(0, test_edges->getDim0()).to(torch::kCPU)});
        }

        auto sorted_edges_accessor = sorted_edges.accessor<int, 2>();
        for (int64_t i = 0; i < sorted_edges.size(0); i++) {
            src_map_.emplace(make_pair(sorted_edges_accessor[i][0], sorted_edges_accessor[i][1]), vector<int>());
            dst_map_.emplace(make_pair(sorted_edges_accessor[i][2], sorted_edges_accessor[i][1]), vector<int>());
        }
        for (int64_t i = 0; i < sorted_edges.size(0); i++) {
            src_map_.at(make_pair(sorted_edges_accessor[i][0], sorted_edges_accessor[i][1])).emplace_back(sorted_edges_accessor[i][2]);
            dst_map_.at(make_pair(sorted_edges_accessor[i][2], sorted_edges_accessor[i][1])).emplace_back(sorted_edges_accessor[i][0]);
        }
        train_edges_->unload(false);
        validation_edges_->unload(false);
        test_edges_->unload(false);
    }

    // If using the partition buffer, we convert to an InMemory storage object.
    // We do this to enable the ability of sampling negatives globally rather than from within each partition
    if (marius_options.storage.embeddings == BackendType::PartitionBuffer) {
        node_embeddings_ = new InMemory(marius_options.path.experiment_directory
                                       + PathConstants::embeddings_directory
                                       + PathConstants::embeddings_file
                                       + PathConstants::file_ext,
                                   embeddings->getDim0(),
                                   marius_options.model.embedding_size,
                                   marius_options.storage.embeddings_dtype, torch::kCPU);
    } else {
        node_embeddings_ = embeddings;
    }
    num_nodes_ = embeddings->getDim0();
    src_relations_ = src_relations;
    dst_relations_ = dst_relations;
    num_relations_ = src_relations_->getDim0();

    initializeBatches();
    batch_iterator_ = batches_.begin();
    timestamp_ = global_timestamp_allocator.getTimestamp();
}

DataSet::DataSet(Storage *test_edges, Storage *embeddings, Storage *src_relations, Storage *dst_relations) {

    train_ = false;
    current_edge_ = 0;
    current_negative_id_ = 0;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    batch_lock_ = new std::mutex();
    negative_lock_ = new std::mutex();
    storage_loaded_ = false;

    test_edges_ = test_edges;
    edges_ = test_edges;
    num_edges_ = edges_->getDim0();

    if (marius_options.evaluation.filtered_evaluation) {
        SPDLOG_ERROR("Filtered MRR requires supplying train and test edges");
        exit(-1);
    }


    if (marius_options.storage.embeddings == BackendType::PartitionBuffer) {
        node_embeddings_ = new InMemory(marius_options.path.experiment_directory
                                       + PathConstants::embeddings_directory
                                       + PathConstants::embeddings_file
                                       + PathConstants::file_ext,
                                   embeddings->getDim0(),
                                   marius_options.model.embedding_size,
                                   marius_options.storage.embeddings_dtype, torch::kCPU);
    } else {
        node_embeddings_ = embeddings;
    }
    num_nodes_ = embeddings->getDim0();
    src_relations_ = src_relations;
    dst_relations_ = dst_relations;

    initializeBatches();
    batch_iterator_ = batches_.begin();
    timestamp_ = global_timestamp_allocator.getTimestamp();
}

DataSet::~DataSet() {
    clearBatches();
    delete batch_lock_;
    delete negative_lock_;
}

void DataSet::initializeBatches() {
    int64_t batch_size = 0;
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    clearBatches();

    vector<Batch *> batches;
    if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
        edge_bucket_sizes_ = edges_->getEdgeBucketSizes();
        for (auto iter = edge_bucket_sizes_.begin(); iter != edge_bucket_sizes_.end(); iter++) {
            batch_size = *iter;
            PartitionBatch *curr_batch = new PartitionBatch(train_);
            curr_batch->batch_id_ = batch_id;
            curr_batch->start_idx_ = start_idx;
            curr_batch->batch_size_ = batch_size;

            curr_batch->src_partition_idx_ = floor(batch_id / sqrt(edge_bucket_sizes_.size()));
            curr_batch->dst_partition_idx_ = batch_id % (int) sqrt(edge_bucket_sizes_.size());

            batches.emplace_back(curr_batch);
            batch_id++;
            start_idx += batch_size;
        }

        batches_ = batches;
        auto ordered_batches = applyOrdering(batches_);
        batches_ = ordered_batches;
        splitBatches();
    } else {
        batch_size = marius_options.training.batch_size;
        if (!train_) {
            batch_size = marius_options.evaluation.batch_size;
        }
        while(start_idx < num_edges_) {
            if (num_edges_ - (start_idx + batch_size) < 0) {
                batch_size = num_edges_ - start_idx;
            }
            Batch *curr_batch = new Batch(train_);
            curr_batch->batch_id_ = batch_id;
            curr_batch->start_idx_ = start_idx;
            curr_batch->batch_size_ = batch_size;

            batches.emplace_back(curr_batch);
            batch_id++;
            start_idx += batch_size;
        }
        batches_ = batches;
    }
}

void DataSet::splitBatches() {
    std::vector<Batch *> new_batches;

    int64_t edge_part_id = 0;
    int64_t batch_id = 0;
    int64_t start_idx = 0;
    for (auto iter = batches_.begin(); iter != batches_.end(); iter++) {
        PartitionBatch *batch = (PartitionBatch *) *iter;
        int64_t batch_size = batch->batch_size_;
        int64_t num_left = batch_size;
        int64_t max_batch_size;
        if (batch->train_) {
            max_batch_size = marius_options.training.batch_size;
        } else {
            max_batch_size = marius_options.evaluation.batch_size;
        }

        if (num_left > 2 * max_batch_size) {
            batch_size = max_batch_size;
        } else if (num_left > max_batch_size) {
            batch_size = num_left / 2;
        } else {
            batch_size = num_left;
        }

        edge_part_id++;

        start_idx = batch->start_idx_;

        while (num_left > 0) {
            PartitionBatch *curr_batch = new PartitionBatch(true);
            curr_batch->batch_id_ = batch_id;
            curr_batch->start_idx_ = start_idx;
            curr_batch->batch_size_ = batch_size;
            curr_batch->src_partition_idx_ = batch->src_partition_idx_;
            curr_batch->dst_partition_idx_ = batch->dst_partition_idx_;
            curr_batch->train_ = train_;

            new_batches.emplace_back(curr_batch);
            batch_id++;
            start_idx += batch_size;

            num_left -= batch_size;
            if (num_left > 2 * max_batch_size) {
                batch_size = max_batch_size;
            } else if (num_left > max_batch_size) {
                batch_size = num_left / 2;
            } else {
                batch_size = num_left;
            }
        }
        delete batch;
    }
    batches_ = new_batches;
}

void DataSet::clearBatches() {
    for (auto iter = batches_.begin(); iter != batches_.end(); ++iter) {
        if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
            PartitionBatch *b = (PartitionBatch *) *iter;
            delete b;
        } else {
            Batch *b = (Batch *) *iter;
            delete b;
        }

    }
    batches_ = std::vector<Batch *>();
}

Batch *DataSet::getBatch() {
    Batch *batch = nextBatch();
    if (batch == nullptr) {
        return batch;
    }

    SPDLOG_TRACE("Starting Batch. ID {}, Starting Index {}, Batch Size {} ", batch->batch_id_, batch->start_idx_, batch->batch_size_);
    globalSample(batch);

    batch->accumulateUniqueIndices();

    if (!train_ && marius_options.evaluation.filtered_evaluation) {
        setEvalFilter(batch);
    }

    loadCPUParameters(batch);

    return batch;
}

Batch *DataSet::nextBatch() {
    std::unique_lock batch_lock(*batch_lock_);
    Batch *batch;
    if (batch_iterator_ != batches_.end()) {
        batch = *batch_iterator_;
        batch_iterator_++;
    } else {
        batch_lock.unlock();
        return nullptr;
    }
    current_edge_ += batch->batch_size_;
    batch_lock.unlock();
    return batch;
}

void DataSet::setEvalFilter(Batch *batch) {
    auto temp_src = batch->src_pos_indices_.to(torch::kCPU);
    auto temp_rel = batch->rel_indices_.to(torch::kCPU);
    auto temp_dst = batch->dst_pos_indices_.to(torch::kCPU);
    auto src_pos_ind_accessor = temp_src.accessor<int64_t, 1>();
    auto rel_ind_accessor = temp_rel.accessor<int64_t, 1>();
    auto dst_pos_ind_accessor = temp_dst.accessor<int64_t, 1>();

    batch->src_neg_filter_eval_ = std::vector<torch::Tensor>(batch->batch_size_);
    batch->dst_neg_filter_eval_ = std::vector<torch::Tensor>(batch->batch_size_);
    for (int64_t i = 0; i < batch->batch_size_; i++) {
        auto src_map_vec = src_map_.at(make_pair(src_pos_ind_accessor[i], rel_ind_accessor[i]));
        auto dst_map_vec = dst_map_.at(make_pair(dst_pos_ind_accessor[i], rel_ind_accessor[i]));
        auto src_map_tens = torch::tensor(src_map_vec);
        auto dst_map_tens = torch::tensor(dst_map_vec);
        batch->src_neg_filter_eval_[i] = dst_map_tens;
        batch->dst_neg_filter_eval_[i] = src_map_tens;
    }
}

void DataSet::globalSample(Batch *batch) {
    EdgeList pos_batch = edges_->range(batch->start_idx_, batch->batch_size_).clone().to(torch::kInt64);
    batch->src_pos_indices_ = pos_batch.select(1, 0);
    batch->dst_pos_indices_ = pos_batch.select(1, 2);
    batch->rel_indices_ = pos_batch.select(1, 1);

    std::unique_lock curr_neg_lock(*negative_lock_);
    // need sample src and dst negs together
    if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
        int64_t src_partition_idx = ((PartitionBatch *) batch)->src_partition_idx_;
        int64_t dst_partition_idx = ((PartitionBatch *) batch)->dst_partition_idx_;
        if (marius_options.training.negative_sampling_access == NegativeSamplingAccess::UniformCrossPartition) {
            batch->src_neg_indices_ = getNegativesIndices().flatten(0, 1);
            batch->dst_neg_indices_ = getNegativesIndices().flatten(0, 1);
        } else {
            batch->src_neg_indices_ = getNegativesIndices(src_partition_idx).flatten(0, 1);
            batch->dst_neg_indices_ = getNegativesIndices(dst_partition_idx).flatten(0, 1);
        }
    } else {
        batch->src_neg_indices_ = getNegativesIndices().flatten(0, 1);
        batch->dst_neg_indices_ = getNegativesIndices().flatten(0, 1);
    }
    curr_neg_lock.unlock();
}

void DataSet::loadCPUParameters(Batch *batch) {
    torch::TensorOptions emb_opts = torch::TensorOptions().dtype(torch::kFloat32);
    // load node embeddings for partitioned ordering
    if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
        int64_t src_partition_idx = ((PartitionBatch *) batch)->src_partition_idx_;
        int64_t dst_partition_idx = ((PartitionBatch *) batch)->dst_partition_idx_;
        if (src_partition_idx == dst_partition_idx) {
            if (marius_options.training.negative_sampling_access == NegativeSamplingAccess::Uniform) {
                batch->unique_node_embeddings_ = ((PartitionBufferStorage *) node_embeddings_)->indexRead(src_partition_idx, batch->unique_node_indices_, 2 * batch->batch_id_);
                SPDLOG_TRACE("Batch: {} Node Embeddings read", batch->batch_id_);
                if (train_) {
                    batch->unique_node_embeddings_state_ = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(src_partition_idx, batch->unique_node_indices_, 2 * batch->batch_id_);
                    SPDLOG_TRACE("Batch: {} Node State read", batch->batch_id_);
                }
            } else {
                Embeddings pos_embs = ((PartitionBufferStorage *) node_embeddings_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->pos_uniques_idx_, 2 * batch->batch_id_);
                Embeddings neg_embs;

                std::tie(((PartitionBatch *) batch)->buffer_state_, neg_embs) = ((PartitionBufferStorage *) node_embeddings_)->bufferIndexRead(((PartitionBatch *) batch)->neg_uniques_idx_);

                if (train_) {
                    OptimizerState pos_embs_state = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->pos_uniques_idx_, 2 * batch->batch_id_);
                    OptimizerState neg_embs_state;
                    std::tie(((PartitionBatch *) batch)->buffer_state_, neg_embs_state) = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->bufferIndexRead(((PartitionBatch *) batch)->neg_uniques_idx_);
                    batch->unique_node_embeddings_ = torch::empty({pos_embs.size(0) + neg_embs.size(0), marius_options.model.embedding_size}, emb_opts);
                    batch->unique_node_embeddings_.narrow(0, 0, pos_embs.size(0)) = pos_embs;
                    batch->unique_node_embeddings_.narrow(0, pos_embs.size(0), neg_embs.size(0)) = neg_embs;
                    batch->unique_node_embeddings_state_ = torch::empty_like(batch->unique_node_embeddings_);
                    batch->unique_node_embeddings_state_.narrow(0, 0, pos_embs_state.size(0)) = pos_embs_state;
                    batch->unique_node_embeddings_state_.narrow(0, pos_embs_state.size(0), neg_embs_state.size(0)) = neg_embs_state;
                } else {
                    batch->unique_node_embeddings_ = torch::empty({pos_embs.size(0) + neg_embs.size(0), marius_options.model.embedding_size}, emb_opts);
                    batch->unique_node_embeddings_.narrow(0, 0, pos_embs.size(0)) = pos_embs;
                    batch->unique_node_embeddings_.narrow(0, pos_embs.size(0), neg_embs.size(0)) = neg_embs;
                }
            }
        } else {
            if (marius_options.training.negative_sampling_access == NegativeSamplingAccess::Uniform) {
                Embeddings src_pos_embs = ((PartitionBufferStorage *) node_embeddings_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->src_pos_uniques_idx_, 2 * batch->batch_id_);
                Embeddings dst_pos_embs = ((PartitionBufferStorage *) node_embeddings_)->indexRead(dst_partition_idx, ((PartitionBatch *) batch)->dst_pos_uniques_idx_, 2 * batch->batch_id_ + 1);
                SPDLOG_TRACE("Batch: {} Node Embeddings read", batch->batch_id_);

                batch->unique_node_embeddings_ = torch::empty({src_pos_embs.size(0) + dst_pos_embs.size(0), marius_options.model.embedding_size}, emb_opts);
                batch->unique_node_embeddings_.narrow(0, 0, src_pos_embs.size(0)) = src_pos_embs;
                batch->unique_node_embeddings_.narrow(0, src_pos_embs.size(0), dst_pos_embs.size(0)) = dst_pos_embs;

                if (train_) {
                    OptimizerState src_pos_embs_state = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->src_pos_uniques_idx_, 2 * batch->batch_id_);
                    OptimizerState dst_pos_embs_state = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(dst_partition_idx, ((PartitionBatch *) batch)->dst_pos_uniques_idx_, 2 * batch->batch_id_ + 1);
                    SPDLOG_TRACE("Batch: {} Node State read", batch->batch_id_);

                    batch->unique_node_embeddings_state_ = torch::empty_like(batch->unique_node_embeddings_);
                    batch->unique_node_embeddings_state_.narrow(0, 0, src_pos_embs_state.size(0)) = src_pos_embs_state;
                    batch->unique_node_embeddings_state_.narrow(0, src_pos_embs_state.size(0), dst_pos_embs_state.size(0)) = dst_pos_embs_state;
                }
            } else {
                Embeddings src_pos_embs = ((PartitionBufferStorage *) node_embeddings_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->src_pos_uniques_idx_, 2 * batch->batch_id_);
                Embeddings dst_pos_embs = ((PartitionBufferStorage *) node_embeddings_)->indexRead(dst_partition_idx, ((PartitionBatch *) batch)->dst_pos_uniques_idx_, 2 * batch->batch_id_ + 1);
                Embeddings neg_embs;
                std::tie(((PartitionBatch *) batch)->buffer_state_, neg_embs) = ((PartitionBufferStorage *) node_embeddings_)->bufferIndexRead(((PartitionBatch *) batch)->neg_uniques_idx_);

                if (train_) {
                    OptimizerState src_pos_embs_state = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(src_partition_idx, ((PartitionBatch *) batch)->src_pos_uniques_idx_, 2 * batch->batch_id_);
                    OptimizerState dst_pos_embs_state = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexRead(dst_partition_idx, ((PartitionBatch *) batch)->dst_pos_uniques_idx_, 2 * batch->batch_id_ + 1);
                    OptimizerState neg_embs_state;
                    std::tie(((PartitionBatch *) batch)->buffer_state_, neg_embs_state) = ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->bufferIndexRead(((PartitionBatch *) batch)->neg_uniques_idx_);
                    batch->unique_node_embeddings_ = torch::empty({src_pos_embs.size(0) + dst_pos_embs.size(0) + neg_embs.size(0), marius_options.model.embedding_size}, emb_opts);
                    batch->unique_node_embeddings_.narrow(0, 0, src_pos_embs.size(0)) = src_pos_embs;
                    batch->unique_node_embeddings_.narrow(0, src_pos_embs.size(0), dst_pos_embs.size(0)) = dst_pos_embs;
                    batch->unique_node_embeddings_.narrow(0, src_pos_embs.size(0) + dst_pos_embs.size(0), neg_embs.size(0)) = neg_embs;
                    batch->unique_node_embeddings_state_ = torch::empty_like(batch->unique_node_embeddings_);
                    batch->unique_node_embeddings_state_.narrow(0, 0, src_pos_embs_state.size(0)) = src_pos_embs_state;
                    batch->unique_node_embeddings_state_.narrow(0, src_pos_embs_state.size(0), dst_pos_embs_state.size(0)) = dst_pos_embs_state;
                    batch->unique_node_embeddings_state_.narrow(0, src_pos_embs_state.size(0) + dst_pos_embs_state.size(0), neg_embs_state.size(0)) = neg_embs_state;
                } else {
                    batch->unique_node_embeddings_ = torch::empty({src_pos_embs.size(0) + dst_pos_embs.size(0) + neg_embs.size(0), marius_options.model.embedding_size}, emb_opts);
                    batch->unique_node_embeddings_.narrow(0, 0, src_pos_embs.size(0)) = src_pos_embs;
                    batch->unique_node_embeddings_.narrow(0, src_pos_embs.size(0), dst_pos_embs.size(0)) = dst_pos_embs;
                    batch->unique_node_embeddings_.narrow(0, src_pos_embs.size(0) + dst_pos_embs.size(0), neg_embs.size(0)) = neg_embs;
                }
            }
        }
        SPDLOG_TRACE("Batch: {} Node embeddings read", batch->batch_id_);

    } else if (marius_options.storage.embeddings != BackendType::DeviceMemory) {
        // load only if embeddings are located on host device
        batch->unique_node_embeddings_ = node_embeddings_->indexRead(batch->unique_node_indices_);
        SPDLOG_TRACE("Batch: {} Node Embeddings read", batch->batch_id_);
        if (train_) {
            batch->unique_node_embeddings_state_ = node_embeddings_optimizer_state_->indexRead(batch->unique_node_indices_);
            SPDLOG_TRACE("Batch: {} Node State read", batch->batch_id_);
        }
    }

    // load relation embeddings (only if stored on host)
    if (marius_options.storage.relations != BackendType::DeviceMemory) {
        if (marius_options.general.num_relations > 1) {
            batch->unique_relation_embeddings_ = torch::stack({src_relations_->indexRead(batch->unique_relation_indices_), dst_relations_->indexRead(batch->unique_relation_indices_)});
            SPDLOG_TRACE("Batch: {} Relation Embeddings read", batch->batch_id_);
            if (train_) {
                batch->unique_relation_embeddings_state_ = torch::stack({src_relations_optimizer_state_->indexRead(batch->unique_relation_indices_), dst_relations_optimizer_state_->indexRead(batch->unique_relation_indices_)});
                SPDLOG_TRACE("Batch: {} Relation State read", batch->batch_id_);
            }
        }
    }

    batch->status_ = BatchStatus::LoadedEmbeddings;
    batch->load_timestamp_ = timestamp_;
}

void DataSet::loadGPUParameters(Batch *batch) {

    if (marius_options.storage.embeddings == BackendType::DeviceMemory) {
            batch->unique_node_embeddings_ = node_embeddings_->indexRead(batch->unique_node_indices_);
            SPDLOG_TRACE("Batch: {} Node Embeddings read", batch->batch_id_);
            if (train_) {
                batch->unique_node_embeddings_state_ = node_embeddings_optimizer_state_->indexRead(batch->unique_node_indices_);
                SPDLOG_TRACE("Batch: {} Node State read", batch->batch_id_);
            }
    }

    if (marius_options.storage.relations == BackendType::DeviceMemory) {
        if (marius_options.general.num_relations > 1) {
            batch->unique_relation_embeddings_ = torch::stack({src_relations_->indexRead(batch->unique_relation_indices_), dst_relations_->indexRead(batch->unique_relation_indices_)});
            SPDLOG_TRACE("Batch: {} Relation Embeddings read", batch->batch_id_);
            if (train_) {
                batch->unique_relation_embeddings_state_ = torch::stack({src_relations_optimizer_state_->indexRead(batch->unique_relation_indices_), dst_relations_optimizer_state_->indexRead(batch->unique_relation_indices_)});
                SPDLOG_TRACE("Batch: {} Relation State read", batch->batch_id_);
            }
        }
    }
}

void DataSet::updateEmbeddingsForBatch(Batch *batch, bool gpu) {
    if (gpu) {
        if (marius_options.storage.embeddings == BackendType::DeviceMemory) {
            node_embeddings_->indexAdd(batch->unique_node_indices_, batch->unique_node_gradients_);
            node_embeddings_optimizer_state_->indexAdd(batch->unique_node_indices_, batch->unique_node_gradients2_);
            SPDLOG_TRACE("Batch: {} Node Embeddings updated", batch->batch_id_);
        }

        if (marius_options.storage.relations == BackendType::DeviceMemory) {
            if (marius_options.general.num_relations > 1) {
                src_relations_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients_.select(0, 0));
                src_relations_optimizer_state_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients2_.select(0, 0));
                dst_relations_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients_.select(0, 1));
                dst_relations_optimizer_state_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients2_.select(0, 1));
                SPDLOG_TRACE("Batch: {} Relation Embeddings updated", batch->batch_id_);
            }
        }
    } else {
        batch->host_transfer_.synchronize();
        if (marius_options.storage.embeddings != BackendType::DeviceMemory) {
            if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
                int64_t src_partition_idx = ((PartitionBatch *) batch)->src_partition_idx_;
                int64_t dst_partition_idx = ((PartitionBatch *) batch)->dst_partition_idx_;
                if (src_partition_idx == dst_partition_idx) {
                    if (marius_options.training.negative_sampling_access == NegativeSamplingAccess::Uniform) {
                        ((PartitionBufferStorage *) node_embeddings_)->indexAdd(src_partition_idx, batch->unique_node_indices_, batch->unique_node_gradients_.narrow(0, 0, batch->unique_node_indices_.size(0)));
                        ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexAdd(src_partition_idx, batch->unique_node_indices_, batch->unique_node_gradients2_.narrow(0, 0, batch->unique_node_indices_.size(0)));
                    } else {
                        Indices pos_idx = ((PartitionBatch *) batch)->pos_uniques_idx_;
                        ((PartitionBufferStorage *) node_embeddings_)->indexAdd(src_partition_idx, pos_idx, batch->unique_node_gradients_.narrow(0, 0, pos_idx.size(0)));
                        ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexAdd(src_partition_idx, pos_idx, batch->unique_node_gradients2_.narrow(0, 0, pos_idx.size(0)));
                    }
                } else {
                    Indices src_partition_emb_inds = ((PartitionBatch *) batch)->src_pos_uniques_idx_;
                    Indices dst_partition_emb_inds = ((PartitionBatch *) batch)->dst_pos_uniques_idx_;

                    Gradients src_partition_emb_grads = batch->unique_node_gradients_.narrow(0, 0, src_partition_emb_inds.size(0));
                    Gradients dst_partition_emb_grads = batch->unique_node_gradients_.narrow(0, src_partition_emb_inds.size(0), dst_partition_emb_inds.size(0));

                    Gradients src_partition_emb_grads2 = batch->unique_node_gradients2_.narrow(0, 0, src_partition_emb_inds.size(0));
                    Gradients dst_partition_emb_grads2 = batch->unique_node_gradients2_.narrow(0, src_partition_emb_inds.size(0), dst_partition_emb_inds.size(0));

                    ((PartitionBufferStorage *) node_embeddings_)->indexAdd(src_partition_idx, src_partition_emb_inds, src_partition_emb_grads);
                    ((PartitionBufferStorage *) node_embeddings_)->indexAdd(dst_partition_idx, dst_partition_emb_inds, dst_partition_emb_grads);
                    ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexAdd(src_partition_idx, src_partition_emb_inds, src_partition_emb_grads2);
                    ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->indexAdd(dst_partition_idx, dst_partition_emb_inds, dst_partition_emb_grads2);
                }

                Indices src_neg_idx = ((PartitionBatch *) batch)->src_neg_indices_;
                Indices dst_neg_idx = ((PartitionBatch *) batch)->dst_neg_indices_;
                Indices neg_idx = ((PartitionBatch *) batch)->neg_uniques_idx_;

                if (marius_options.training.negative_sampling_access == NegativeSamplingAccess::UniformCrossPartition) {
                    Gradients neg_grads = batch->unique_node_gradients_.narrow(0, ((PartitionBatch *) batch)->pos_uniques_idx_.size(0), neg_idx.size(0));
                    Gradients neg_grads2 = batch->unique_node_gradients2_.narrow(0, ((PartitionBatch *) batch)->pos_uniques_idx_.size(0), neg_idx.size(0));

                    ((PartitionBufferStorage *) node_embeddings_)->bufferIndexAdd(((PartitionBatch *) batch)->buffer_state_, neg_idx, neg_grads);
                    ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->bufferIndexAdd(((PartitionBatch *) batch)->buffer_state_, neg_idx, neg_grads2);
                }
            } else {
                node_embeddings_->indexAdd(batch->unique_node_indices_, batch->unique_node_gradients_);
                node_embeddings_optimizer_state_->indexAdd(batch->unique_node_indices_, batch->unique_node_gradients2_);
                SPDLOG_TRACE("Batch: {} Node Embeddings updated", batch->batch_id_);
            }
        }

        if (marius_options.storage.relations != BackendType::DeviceMemory) {
            if (marius_options.general.num_relations > 1) {
                src_relations_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients_.select(0, 0));
                src_relations_optimizer_state_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients2_.select(0, 0));
                dst_relations_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients_.select(0, 1));
                dst_relations_optimizer_state_->indexAdd(batch->unique_relation_indices_, batch->unique_relation_gradients2_.select(0, 1));
                SPDLOG_TRACE("Batch: {} Relation Embeddings updated", batch->batch_id_);
            }
        }
        batch->clear();
    }
}

Indices DataSet::shuffleIndices() {
    int num_chunks = marius_options.training.number_of_chunks;
    int negatives = marius_options.training.negatives;
    NegativeSamplingAccess negative_sampling_access = marius_options.training.negative_sampling_access;

    if (!train_) {
        num_chunks = marius_options.evaluation.number_of_chunks;
        negatives = marius_options.evaluation.negatives;
        negative_sampling_access = marius_options.evaluation.negative_sampling_access;
    }

    if (negative_sampling_access == NegativeSamplingAccess::All) {
        current_negative_id_ = 0;
        num_chunks = 1;
        negatives = num_nodes_;
    }

    int64_t rand_left;
    vector<Indices> ret_indices(num_chunks);
    int64_t current_negative_id = current_negative_id_;
    current_negative_id_ = (current_negative_id_ + (num_chunks * negatives)) % num_nodes_;

    torch::TensorOptions ind_opts;

    if (marius_options.storage.embeddings == BackendType::DeviceMemory) {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(marius_options.general.device);
    } else {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    }

    for (int j = 0; j < num_chunks; j++) {
        int num_passes = (int) ceil((float) (current_negative_id + negatives) / num_nodes_);
        Embeddings negatives_data;
        Indices indices;
        if (num_passes == 1) {
            indices = torch::arange((int64_t) current_negative_id, (int64_t) current_negative_id + negatives, ind_opts);
            current_negative_id += negatives;
            current_negative_id = current_negative_id % num_nodes_;
        } else {
            int64_t curr_idx = 0;
            int64_t num_itr = 0;
            int64_t negs_left = negatives;
            indices = torch::empty({negatives}, ind_opts);

            while (negs_left > 0) {
                rand_left = num_nodes_ - current_negative_id;
                if (negs_left < rand_left) {
                    num_itr = negs_left;
                } else {
                    num_itr = rand_left;
                }
                indices.narrow(0, curr_idx, num_itr) = torch::arange((int64_t) current_negative_id, (int64_t) current_negative_id + num_itr, ind_opts);

                negs_left -= num_itr;
                curr_idx += num_itr;

                current_negative_id += num_itr;
                current_negative_id = current_negative_id % num_nodes_;
            }
        }
        ret_indices[j] = indices;
    }
    return torch::stack(ret_indices);
}

Indices DataSet::uniformIndices() {
    int num_chunks = marius_options.training.number_of_chunks;
    int negatives = marius_options.training.negatives * (1 - marius_options.training.degree_fraction);

    if (!train_) {
        num_chunks = marius_options.evaluation.number_of_chunks;
        negatives = marius_options.evaluation.negatives * (1 - marius_options.evaluation.degree_fraction);
    }

    vector<Indices> ret_indices(num_chunks);
    Indices rand_idx;
    int64_t num_nodes = num_nodes_;

    if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
        num_nodes = ((PartitionBufferStorage *) node_embeddings_)->getBufferEmbeddingsCapacity();
    }

    torch::TensorOptions ind_opts;
    if (marius_options.storage.edges == BackendType::DeviceMemory) {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(marius_options.general.device);
    } else {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    }

    for (int j = 0; j < num_chunks; j++) {
        rand_idx = torch::randint(0, num_nodes, {negatives}, ind_opts);
        ret_indices[j] = rand_idx;
    }

    Indices ret_ind = torch::stack(ret_indices);

    return torch::stack(ret_indices);
}

Indices DataSet::uniformIndices(int partition_id) {
    int num_chunks = marius_options.training.number_of_chunks;
    int negatives = marius_options.training.negatives * (1 - marius_options.training.degree_fraction);

    if (!train_) {
        num_chunks = marius_options.evaluation.number_of_chunks;
        negatives = marius_options.evaluation.negatives * (1 - marius_options.evaluation.degree_fraction);
    }

    vector<Indices> ret_indices(num_chunks);
    Indices rand_idx;
    int64_t num_nodes = num_nodes_;

    int64_t offset = partition_id * ((PartitionBufferStorage *) node_embeddings_)->getPartitionSize();

    if (marius_options.storage.embeddings == BackendType::PartitionBuffer && train_) {
        num_nodes = ((PartitionBufferStorage *) node_embeddings_)->getPartitionSize();
        if (partition_id == marius_options.storage.num_partitions - 1) {
            num_nodes = ((PartitionBufferStorage *) node_embeddings_)->getDim0() - partition_id * ((PartitionBufferStorage *) node_embeddings_)->getPartitionSize();
        }
    }

    torch::TensorOptions ind_opts;

    if (marius_options.storage.edges == BackendType::DeviceMemory) {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(marius_options.general.device);
    } else {
        ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    }

    for (int j = 0; j < num_chunks; j++) {
        rand_idx = torch::randint(0, num_nodes, {negatives}, ind_opts);
        ret_indices[j] = rand_idx;
    }

    Indices ret_ind = torch::stack(ret_indices);

    return torch::stack(ret_indices) + offset;
}

Indices DataSet::getNegativesIndices() {
    Indices res;

    NegativeSamplingAccess negative_sampling_access = marius_options.training.negative_sampling_access;

    if (!train_) {
        negative_sampling_access = marius_options.evaluation.negative_sampling_access;
    }

    switch (negative_sampling_access) {
        case NegativeSamplingAccess::Uniform:
            res = uniformIndices();
            break;
        case NegativeSamplingAccess::UniformCrossPartition:
            res = uniformIndices();
            break;
        case NegativeSamplingAccess::All:
            res = shuffleIndices();
            break;
    }
    return res;
}

Indices DataSet::getNegativesIndices(int partition_id) {
    return uniformIndices(partition_id);
}

void DataSet::nextEpoch() {
    if (train_) {
        if(((epochs_processed_ + 1) % marius_options.training.checkpoint_interval) == 0) {
            checkpointParameters();
        }

        if(((epochs_processed_ + 1) % marius_options.training.shuffle_interval) == 0) {
            edges_->shuffle();
            SPDLOG_INFO("Edges Shuffled");
        }

        if (marius_options.storage.embeddings == BackendType::PartitionBuffer) {
            ((PartitionBufferStorage *) node_embeddings_)->sync();
            ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->sync();
            initializeBatches();
            batches_ = ((PartitionBufferStorage *) node_embeddings_)->shuffleBeforeEvictions(batches_);
            ((PartitionBufferStorage *) node_embeddings_)->setOrdering(batches_);
            ((PartitionBufferStorage *) node_embeddings_optimizer_state_)->setOrdering(batches_);
        }
    }

    current_edge_ = 0;
    batches_processed_ = 0;
    epochs_processed_++;
    batch_iterator_ = batches_.begin();
}

void DataSet::checkpointParameters() {
    node_embeddings_->checkpoint(epochs_processed_);
    src_relations_->checkpoint(epochs_processed_);
    dst_relations_->checkpoint(epochs_processed_);
}

void DataSet::loadStorage() {
    edges_->load();
    if (train_) {
        node_embeddings_->load();
        node_embeddings_optimizer_state_->load();
        src_relations_->load();
        src_relations_optimizer_state_->load();
        dst_relations_->load();
        dst_relations_optimizer_state_->load();
        SPDLOG_DEBUG("Loaded Training Set");
    } else {
        node_embeddings_->load();
        src_relations_->load();
        dst_relations_->load();
        SPDLOG_DEBUG("Loaded Evaluation Set");
    }
    storage_loaded_ = true;
}

void DataSet::unloadStorage() {
    edges_->unload();
    if (train_) {
        node_embeddings_->unload(true);
        node_embeddings_optimizer_state_->unload(true);
        src_relations_->unload(true);
        src_relations_optimizer_state_->unload(true);
        dst_relations_->unload(true);
        dst_relations_optimizer_state_->unload(true);
        SPDLOG_DEBUG("Unloaded Training Set");
    } else {
        node_embeddings_->unload();
        src_relations_->unload();
        dst_relations_->unload();
        SPDLOG_DEBUG("Unloaded Evaluation Set");
    }
    storage_loaded_ = false;
}

torch::Tensor DataSet::accumulateRanks() {
    auto ranks_opts = torch::TensorOptions().dtype(torch::kInt64).device(marius_options.general.device);
    int64_t total_ranks = 0;
    for (auto itr = batches_.begin(); itr != batches_.end(); itr++) {
        total_ranks += (*itr)->ranks_.size(0);
    }
    torch::Tensor ranks = torch::zeros({total_ranks}, ranks_opts);
    int64_t start = 0;
    int64_t length = 0;
    for (auto itr = batches_.begin(); itr != batches_.end(); itr++) {
        length = (*itr)->ranks_.size(0);
        ranks.narrow(0, start, length) = (*itr)->ranks_.clone();
        start += length;
        (*itr)->ranks_ = torch::Tensor(); // free memory
    }

    CudaEvent transfer(0);
    torch::Tensor ret_ranks = ranks.to(torch::kDouble).to(torch::kCPU);

    transfer.record();
    transfer.synchronize();
    return ret_ranks;
}

float DataSet::accumulateAuc() {
    float total_auc = 0;
    for (auto itr = batches_.begin(); itr != batches_.end(); itr++) {
        total_auc += (*itr)->auc_.item<float>();
    }
    return total_auc / (int) batches_.size();
}

