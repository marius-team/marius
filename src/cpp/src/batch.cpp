//
// Created by Jason Mohoney on 7/9/20.
//

#include <batch.h>

using std::get;

Batch::Batch(bool train) : device_transfer_(0), host_transfer_(0), timer_(false) {
    status_ = BatchStatus::Waiting;
    train_ = train;
    device_id_ = -1;
}

void Batch::localSample() {

    int num_deg;
    if (train_) {
        num_deg = (int) (marius_options.training.negatives * marius_options.training.degree_fraction);
    } else {
        num_deg = (int) (marius_options.evaluation.negatives * marius_options.evaluation.degree_fraction);

        if (marius_options.evaluation.negative_sampling_access == NegativeSamplingAccess::All) {
            // no need to sample by degree when using all nodes for sampling negatives
            src_all_neg_embeddings_ = src_global_neg_embeddings_;
            dst_all_neg_embeddings_ = dst_global_neg_embeddings_;
            return;
        }
    }

    if (num_deg == 0) {
        src_all_neg_embeddings_ = src_global_neg_embeddings_;
        dst_all_neg_embeddings_ = dst_global_neg_embeddings_;
        return;
    }

    int num_chunks = src_global_neg_embeddings_.size(0);
    int num_per_chunk = (int) ceil((float) batch_size_ / num_chunks);

    // Sample for src and update filter
    int src_neg_size = src_global_neg_embeddings_.size(1);
    Indices src_sample = torch::randint(0, batch_size_, {num_chunks, num_deg}, src_global_neg_embeddings_.device()).to(torch::kInt64);
    auto chunk_ids = (src_sample.floor_divide(num_per_chunk)).view({num_chunks, -1});
    auto inv_mask = chunk_ids - torch::arange(0, num_chunks, src_global_neg_embeddings_.device()).view({num_chunks, -1});
    auto mask = (inv_mask == 0);
    auto temp_idx = torch::nonzero(mask);
    auto id_offset = src_sample.flatten(0, 1).index_select(0, (temp_idx.select(1, 0) * num_deg + temp_idx.select(1, 1)));
    auto sample_offset = temp_idx.select(1, 1);
    src_all_neg_embeddings_ = torch::cat({src_global_neg_embeddings_, src_pos_embeddings_.index_select(0, src_sample.flatten(0, 1)).view({num_chunks, num_deg, -1})}, 1);
    src_neg_filter_ = id_offset * src_all_neg_embeddings_.size(1) + (src_neg_size + sample_offset);

    // Sample for dst and update filter
    int dst_neg_size = dst_global_neg_embeddings_.size(1);
    Indices dst_sample = torch::randint(0, batch_size_, {num_chunks, num_deg}, dst_global_neg_embeddings_.device()).to(torch::kInt64);
    chunk_ids = (dst_sample.floor_divide(num_per_chunk)).view({num_chunks, -1});
    inv_mask = chunk_ids - torch::arange(0, num_chunks, dst_global_neg_embeddings_.device()).view({num_chunks, -1});
    mask = (inv_mask == 0);
    temp_idx = torch::nonzero(mask);
    id_offset = dst_sample.flatten(0, 1).index_select(0, (temp_idx.select(1, 0) * num_deg + temp_idx.select(1, 1)));
    sample_offset = temp_idx.select(1, 1);
    dst_all_neg_embeddings_ = torch::cat({dst_global_neg_embeddings_, dst_pos_embeddings_.index_select(0, dst_sample.flatten(0, 1)).view({num_chunks, num_deg, -1})}, 1);
    dst_neg_filter_ = id_offset * dst_all_neg_embeddings_.size(1) + (dst_neg_size + sample_offset);
}

void Batch::accumulateUniqueIndices() {
    Indices emb_idx = torch::cat({src_pos_indices_, dst_pos_indices_, src_neg_indices_, dst_neg_indices_});

    auto unique_tup = torch::_unique2(emb_idx, true, true, false);
    unique_node_indices_ = get<0>(unique_tup);
    Indices emb_mapping = get<1>(unique_tup);

    int64_t curr = 0;
    int64_t size = batch_size_;
    src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
    curr += size;
    dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
    curr += size;
    size = src_neg_indices_.size(0);
    src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
    curr += size;
    dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);

    SPDLOG_TRACE("Batch: {} Accumulated {} unique embeddings", batch_id_, unique_node_indices_.size(0));

    if (marius_options.general.num_relations > 1) {
        auto rel_unique_tup = torch::_unique2(rel_indices_, true, true, false);
        unique_relation_indices_ = get<0>(rel_unique_tup);
        rel_indices_mapping_ = get<1>(rel_unique_tup);

        SPDLOG_TRACE("Batch: {} Accumulated {} unique relations", batch_id_, unique_relation_indices_.size(0));
    }
    status_ = BatchStatus::AccumulatedIndices;
}

void Batch::embeddingsToDevice(int device_id) {

    device_id_ = device_id;

    if (marius_options.general.device == torch::kCUDA) {

        device_transfer_ = CudaEvent(device_id);
        host_transfer_ = CudaEvent(device_id);
        string device_string = "cuda:" + std::to_string(device_id);

        src_pos_indices_mapping_ = src_pos_indices_mapping_.to(device_string, true);
        dst_pos_indices_mapping_ = dst_pos_indices_mapping_.to(device_string, true);
        src_neg_indices_mapping_ = src_neg_indices_mapping_.to(device_string, true);
        dst_neg_indices_mapping_ = dst_neg_indices_mapping_.to(device_string, true);
        SPDLOG_TRACE("Batch: {} Indices sent to device", batch_id_);

        if (marius_options.storage.embeddings != BackendType::DeviceMemory) {
            unique_node_embeddings_ = unique_node_embeddings_.to(device_string, true);
            SPDLOG_TRACE("Batch: {} Embeddings sent to device", batch_id_);

            if (train_) {
                unique_node_embeddings_state_ = unique_node_embeddings_state_.to(device_string, true);
                SPDLOG_TRACE("Batch: {} Node State sent to device", batch_id_);
            }
        }

        if (marius_options.general.num_relations > 1) {
            if (marius_options.storage.relations != BackendType::DeviceMemory) {
                unique_relation_embeddings_ = unique_relation_embeddings_.to(device_string, true);
                rel_indices_mapping_ = rel_indices_mapping_.to(device_string, true);
                SPDLOG_TRACE("Batch: {} Relations sent to device", batch_id_);
                if (train_) {
                    unique_relation_embeddings_state_ = unique_relation_embeddings_state_.to(device_string, true);
                    SPDLOG_TRACE("Batch: {} Relation State sent to device", batch_id_);
                }
            } else {
                unique_relation_indices_ = unique_relation_indices_.to(device_string, true);
                rel_indices_mapping_ = rel_indices_mapping_.to(device_string, true);
            }
        }

    }
    device_transfer_.record();
    status_ = BatchStatus::TransferredToDevice;
}

void Batch::prepareBatch() {

    device_transfer_.synchronize();
    int64_t num_chunks = 0;
    int64_t negatives = 0;
    if (train_) {
        num_chunks = marius_options.training.number_of_chunks;
        negatives = marius_options.training.negatives * (1 - marius_options.training.degree_fraction);
    } else {
        num_chunks = marius_options.evaluation.number_of_chunks;
        negatives = marius_options.evaluation.negatives * (1 - marius_options.evaluation.degree_fraction);

        if (marius_options.evaluation.negative_sampling_access == NegativeSamplingAccess::All) {
            num_chunks = 1;
            negatives = marius_options.general.num_nodes;
        }
    }

    src_pos_embeddings_ = unique_node_embeddings_.index_select(0, src_pos_indices_mapping_);
    dst_pos_embeddings_ = unique_node_embeddings_.index_select(0, dst_pos_indices_mapping_);
    src_global_neg_embeddings_ = unique_node_embeddings_.index_select(0, src_neg_indices_mapping_).reshape({num_chunks, negatives, marius_options.model.embedding_size});
    dst_global_neg_embeddings_ = unique_node_embeddings_.index_select(0, dst_neg_indices_mapping_).reshape({num_chunks, negatives, marius_options.model.embedding_size});
    if (marius_options.general.num_relations > 1) {
        src_relation_emebeddings_ = unique_relation_embeddings_.index_select(1, rel_indices_mapping_).select(0, 0);
        dst_relation_emebeddings_ = unique_relation_embeddings_.index_select(1, rel_indices_mapping_).select(0, 1);
    }

    if (train_) {
        src_pos_embeddings_.requires_grad_();
        dst_pos_embeddings_.requires_grad_();
        src_global_neg_embeddings_.requires_grad_();
        dst_global_neg_embeddings_.requires_grad_();
        if (marius_options.general.num_relations > 1) {
            src_relation_emebeddings_.requires_grad_();
            dst_relation_emebeddings_.requires_grad_();
        }
    }

    SPDLOG_TRACE("Batch: {} prepared for compute", batch_id_);
    status_ = BatchStatus::PreparedForCompute;
}

void Batch::accumulateGradients() {

    auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(src_pos_embeddings_.device());

    unique_node_gradients_ = torch::zeros({unique_node_indices_.size(0), marius_options.model.embedding_size}, grad_opts);
    unique_node_gradients_.index_add_(0, src_pos_indices_mapping_, src_pos_embeddings_.grad());
    unique_node_gradients_.index_add_(0, src_neg_indices_mapping_, src_global_neg_embeddings_.grad().flatten(0, 1));
    unique_node_gradients_.index_add_(0, dst_pos_indices_mapping_, dst_pos_embeddings_.grad());
    unique_node_gradients_.index_add_(0, dst_neg_indices_mapping_, dst_global_neg_embeddings_.grad().flatten(0, 1));
    SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);

    if (marius_options.general.num_relations > 1) {
        unique_relation_gradients_ = torch::zeros({2, unique_relation_indices_.size(0), marius_options.model.embedding_size}, grad_opts);
        unique_relation_gradients_.index_add_(1, rel_indices_mapping_, torch::stack({src_relation_emebeddings_.grad(), dst_relation_emebeddings_.grad()}));
        SPDLOG_TRACE("Batch: {} accumulated relation gradients", batch_id_);

        unique_relation_gradients2_ = unique_relation_gradients_.pow(2);
        unique_relation_embeddings_state_.add_(unique_relation_gradients2_);
        unique_relation_gradients_ = -marius_options.training.learning_rate * (unique_relation_gradients_ / ((unique_relation_embeddings_state_).sqrt().add_(1e-9)));
    }

    unique_node_gradients2_ = unique_node_gradients_.pow(2);
    unique_node_embeddings_state_.add_(unique_node_gradients2_);
    unique_node_gradients_ = -marius_options.training.learning_rate * (unique_node_gradients_ / ((unique_node_embeddings_state_).sqrt().add_(1e-9)));
    SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);

    src_pos_embeddings_ = torch::Tensor();
    dst_pos_embeddings_ = torch::Tensor();
    src_global_neg_embeddings_ = torch::Tensor();
    dst_global_neg_embeddings_ = torch::Tensor();
    src_all_neg_embeddings_ = torch::Tensor();
    dst_all_neg_embeddings_ = torch::Tensor();
    src_relation_emebeddings_ = torch::Tensor();
    dst_relation_emebeddings_ = torch::Tensor();

    unique_node_embeddings_state_ = torch::Tensor();
    unique_relation_embeddings_state_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;
}

void Batch::embeddingsToHost() {

    torch::DeviceType emb_device = torch::kCPU;

    if (marius_options.storage.embeddings == BackendType::DeviceMemory) {
        // only single gpu setup
        SPDLOG_TRACE("Batch: {} embedding storage on device", batch_id_);
        emb_device = marius_options.general.device;
    }

    if (emb_device == torch::kCPU && unique_node_gradients_.device().type() == torch::kCUDA) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(unique_node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(unique_node_gradients_, true);
        Gradients temp_grads2 = torch::empty(unique_node_gradients2_.sizes(), grad_opts);
        temp_grads2.copy_(unique_node_gradients2_, true);
        unique_node_gradients_ = temp_grads;
        unique_node_gradients2_ = temp_grads2;
        SPDLOG_TRACE("Batch: {} transferred node embeddings to host", batch_id_);
    }

    if (marius_options.general.num_relations > 1) {
        torch::DeviceType rel_device = torch::kCPU;
        if (marius_options.storage.relations == BackendType::DeviceMemory) {
            SPDLOG_TRACE("Batch: {} relation storage on device", batch_id_);
            rel_device = marius_options.general.device;
        }
        if (rel_device == torch::kCPU && unique_relation_gradients_.device().type() == torch::kCUDA) {
            auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
            Gradients temp_grads = torch::empty(unique_relation_gradients_.sizes(), grad_opts);
            temp_grads.copy_(unique_relation_gradients_, true);
            Gradients temp_grads2 = torch::empty(unique_relation_gradients2_.sizes(), grad_opts);
            temp_grads2.copy_(unique_relation_gradients2_, true);
            unique_relation_gradients_ = temp_grads;
            unique_relation_gradients2_ = temp_grads2;
            SPDLOG_TRACE("Batch: {} transferred relation embeddings to host", batch_id_);
        }
    }
    host_transfer_.record();
    host_transfer_.synchronize();
    status_ = BatchStatus::TransferredToHost;
}

void Batch::clear() {
    unique_node_indices_ = torch::Tensor();
    unique_node_embeddings_ = torch::Tensor();
    unique_node_gradients_ = torch::Tensor();
    unique_node_gradients2_ = torch::Tensor();
    unique_node_embeddings_state_ = torch::Tensor();

    unique_relation_indices_ = torch::Tensor();
    unique_relation_embeddings_ = torch::Tensor();
    unique_relation_gradients_ = torch::Tensor();
    unique_relation_gradients2_ = torch::Tensor();
    unique_relation_embeddings_state_ = torch::Tensor();

    src_pos_indices_mapping_ = torch::Tensor();
    dst_pos_indices_mapping_ = torch::Tensor();
    rel_indices_mapping_ = torch::Tensor();
    src_neg_indices_mapping_ = torch::Tensor();
    dst_neg_indices_mapping_ = torch::Tensor();

    src_pos_indices_ = torch::Tensor();
    dst_pos_indices_ = torch::Tensor();
    rel_indices_ = torch::Tensor();
    src_neg_indices_ = torch::Tensor();
    dst_neg_indices_ = torch::Tensor();

    src_pos_embeddings_ = torch::Tensor();
    dst_pos_embeddings_ = torch::Tensor();
    src_relation_emebeddings_ = torch::Tensor();
    dst_relation_emebeddings_ = torch::Tensor();
    src_global_neg_embeddings_ = torch::Tensor();
    dst_global_neg_embeddings_ = torch::Tensor();
    src_all_neg_embeddings_ = torch::Tensor();
    dst_all_neg_embeddings_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared", batch_id_);
    status_ = BatchStatus::Done;
}

PartitionBatch::PartitionBatch(bool train) : Batch(train) {}

// TODO optionally keep unique index data around so it doesn't need to be recomputed in future epochs
void PartitionBatch::clear() {
    Batch::clear();
    pos_uniques_idx_ = torch::Tensor();
    src_pos_uniques_idx_ = torch::Tensor();
    dst_pos_uniques_idx_ = torch::Tensor();
    neg_uniques_idx_ = torch::Tensor();
}

void PartitionBatch::accumulateUniqueIndices() {

    NegativeSamplingAccess negative_sampling_access = marius_options.training.negative_sampling_access;
    if (!train_) {
        negative_sampling_access = marius_options.evaluation.negative_sampling_access;
    }

    if (negative_sampling_access == NegativeSamplingAccess::Uniform) {
        if (src_partition_idx_ == dst_partition_idx_) {
            Indices emb_idx = torch::cat({src_pos_indices_, dst_pos_indices_, src_neg_indices_, dst_neg_indices_});

            auto unique_tup = torch::_unique2(emb_idx, true, true, false);
            unique_node_indices_ = get<0>(unique_tup);
            Indices emb_mapping = get<1>(unique_tup);

            int64_t curr = 0;
            int64_t size = batch_size_;
            src_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            dst_pos_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            size = src_neg_indices_.size(0);
            src_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
            curr += size;
            dst_neg_indices_mapping_ = emb_mapping.narrow(0, curr, size);
        } else {
            Indices src_emb_idx = torch::cat({src_pos_indices_, src_neg_indices_});
            auto src_pos_unique_tup = torch::_unique2(src_emb_idx, true, true, false);
            auto src_pos_uniques_idx = get<0>(src_pos_unique_tup);
            auto src_emb_mapping = get<1>(src_pos_unique_tup);

            int64_t curr = 0;
            int64_t size = batch_size_;
            src_pos_indices_mapping_ = src_emb_mapping.narrow(0, curr, size);
            curr += size;
            size = src_neg_indices_.size(0);
            src_neg_indices_mapping_ = src_emb_mapping.narrow(0, curr, size);

            Indices dst_emb_idx = torch::cat({dst_pos_indices_, dst_neg_indices_});
            auto dst_pos_unique_tup = torch::_unique2(dst_emb_idx, true, true, false);
            auto dst_pos_uniques_idx = get<0>(dst_pos_unique_tup);
            auto dst_emb_mapping = get<1>(dst_pos_unique_tup) + src_pos_uniques_idx.size(0);

            curr = 0;
            size = batch_size_;
            dst_pos_indices_mapping_ = dst_emb_mapping.narrow(0, curr, size);
            curr += size;
            size = dst_neg_indices_.size(0);
            dst_neg_indices_mapping_ = dst_emb_mapping.narrow(0, curr, size);

            unique_node_indices_ = torch::cat({src_pos_uniques_idx, dst_pos_uniques_idx});
            pos_uniques_idx_ = unique_node_indices_.narrow(0, 0, src_pos_uniques_idx.size(0) + dst_pos_uniques_idx.size(0));
            src_pos_uniques_idx_ = unique_node_indices_.narrow(0, 0, src_pos_uniques_idx.size(0));
            dst_pos_uniques_idx_ = unique_node_indices_.narrow(0, src_pos_uniques_idx.size(0), dst_pos_uniques_idx.size(0));
        }
    } else {
        if (src_partition_idx_ == dst_partition_idx_) {
            // calculate uniques for positives
            Indices pos_emb_idx = torch::cat({src_pos_indices_, dst_pos_indices_});
            auto pos_unique_tup = torch::_unique2(pos_emb_idx, true, true, false);
            auto pos_uniques_idx = get<0>(pos_unique_tup);
            Indices pos_emb_mapping = get<1>(pos_unique_tup);
            int64_t curr = 0;
            int64_t size = batch_size_;
            src_pos_indices_mapping_ = pos_emb_mapping.narrow(0, curr, size);
            curr += size;
            dst_pos_indices_mapping_ = pos_emb_mapping.narrow(0, curr, size);

            // calculate uniques for negatives
            Indices neg_uniques_idx;
            Indices neg_emb_idx = torch::cat({src_neg_indices_, dst_neg_indices_});
            auto neg_unique_tup = torch::_unique2(neg_emb_idx, true, true, false);
            neg_uniques_idx = get<0>(neg_unique_tup);
            Indices neg_emb_mapping = get<1>(neg_unique_tup);
            curr = 0;
            size = src_neg_indices_.size(0);
            src_neg_indices_mapping_ = neg_emb_mapping.narrow(0, curr, size);
            curr += size;
            dst_neg_indices_mapping_ = neg_emb_mapping.narrow(0, curr, size);

            // add offset to negative mapping to account for the torch::cat
            src_neg_indices_mapping_ += pos_uniques_idx.size(0);
            dst_neg_indices_mapping_ += pos_uniques_idx.size(0);

            unique_node_indices_ = torch::cat({pos_uniques_idx, neg_uniques_idx});
            pos_uniques_idx_ = unique_node_indices_.narrow(0, 0, pos_uniques_idx.size(0));
            neg_uniques_idx_ = unique_node_indices_.narrow(0, pos_uniques_idx.size(0), neg_uniques_idx.size(0));
        } else {
            auto src_pos_unique_tup = torch::_unique2(src_pos_indices_, true, true, false);
            auto src_pos_uniques_idx = get<0>(src_pos_unique_tup);
            src_pos_indices_mapping_ = get<1>(src_pos_unique_tup);

            auto dst_pos_unique_tup = torch::_unique2(dst_pos_indices_, true, true, false);
            auto dst_pos_uniques_idx = get<0>(dst_pos_unique_tup);
            dst_pos_indices_mapping_ = get<1>(dst_pos_unique_tup) + src_pos_uniques_idx.size(0);

            Indices neg_uniques_idx;
            int64_t size;
            int64_t curr;
            Indices neg_emb_idx = torch::cat({src_neg_indices_, dst_neg_indices_});
            auto neg_unique_tup = torch::_unique2(neg_emb_idx, true, true, false);
            neg_uniques_idx = get<0>(neg_unique_tup);
            Indices neg_emb_mapping = get<1>(neg_unique_tup);
            curr = 0;
            size = src_neg_indices_.size(0);
            src_neg_indices_mapping_ = neg_emb_mapping.narrow(0, curr, size);
            curr += size;
            dst_neg_indices_mapping_ = neg_emb_mapping.narrow(0, curr, size);

            // add offset to negative mapping to account for the torch::cat
            src_neg_indices_mapping_ += src_pos_uniques_idx.size(0) + dst_pos_uniques_idx.size(0);
            dst_neg_indices_mapping_ += src_pos_uniques_idx.size(0) + dst_pos_uniques_idx.size(0);

            unique_node_indices_ = torch::cat({src_pos_uniques_idx, dst_pos_uniques_idx, neg_uniques_idx});
            pos_uniques_idx_ = unique_node_indices_.narrow(0, 0, src_pos_uniques_idx.size(0) + dst_pos_uniques_idx.size(0));
            src_pos_uniques_idx_ = unique_node_indices_.narrow(0, 0, src_pos_uniques_idx.size(0));
            dst_pos_uniques_idx_ = unique_node_indices_.narrow(0, src_pos_uniques_idx.size(0), dst_pos_uniques_idx.size(0));
            neg_uniques_idx_ = unique_node_indices_.narrow(0, src_pos_uniques_idx.size(0) + dst_pos_uniques_idx.size(0), neg_uniques_idx.size(0));
        }
    }

    rel_indices_mapping_ = torch::zeros({1}, torch::kInt64);
    unique_relation_indices_ = torch::zeros({1}, torch::kInt64);

    if (marius_options.general.num_relations > 1) {
        auto rel_unique_tup = torch::_unique2(rel_indices_, true, true, false);
        unique_relation_indices_ = get<0>(rel_unique_tup);
        rel_indices_mapping_ = get<1>(rel_unique_tup);
    }
    status_ = BatchStatus::AccumulatedIndices;
}
