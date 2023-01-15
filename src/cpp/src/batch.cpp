//
// Created by Jason Mohoney on 7/9/20.
//

#include "batch.h"

#include "configuration/constants.h"
#include "logger.h"

using std::get;

Batch::Batch(bool train) : device_transfer_(0), host_transfer_(0), timer_(false) {
    status_ = BatchStatus::Waiting;
    train_ = train;
    device_id_ = -1;
}

Batch::Batch(std::vector<Batch *> sub_batches) : device_transfer_(0), host_transfer_(0), timer_(false) {

    std::vector<torch::Tensor> unique_node_ids_vec;
    std::vector<int> offsets;
    std::vector<int> sizes;

    int offset = 0;
    batch_size_ = 0;
    for (auto batch : sub_batches) {
        unique_node_ids_vec.emplace_back(batch->unique_node_indices_);
        offsets.emplace_back(offset);
        sizes.emplace_back(batch->unique_node_indices_.size(0));
        offset += batch->unique_node_indices_.size(0);
        batch_size_ += batch->batch_size_;
    }

    torch::Tensor all_ids = torch::cat(unique_node_ids_vec);

    auto tup = torch::_unique(all_ids, true, true);
    unique_node_indices_ = std::get<0>(tup);
    torch::Tensor id_mapping = std::get<1>(tup);

    unique_node_gradients_ = torch::zeros({unique_node_indices_.size(0), sub_batches[0]->unique_node_gradients_.size(1)}, torch::kFloat32);
    unique_node_state_update_ = torch::zeros_like(unique_node_gradients_);

    for (int i = 0; i < offsets.size() ; i++) {
        torch::Tensor mapping = id_mapping.narrow(0, offsets[i], sizes[i]);
        unique_node_gradients_.index_add_(0, mapping, sub_batches[i]->unique_node_gradients_);
        unique_node_state_update_.index_add_(0, mapping, sub_batches[i]->unique_node_state_update_);
    }

    for (auto batch : sub_batches) {
        batch->clear();
        delete batch;
    }
}

Batch::~Batch() {
    clear();
}

void Batch::localSample() {

    int num_chunks = negative_sampling_->num_chunks;
    int num_local_negatives = negative_sampling_->negatives_per_positive * negative_sampling_->degree_fraction;

    if (num_local_negatives == 0) {
        src_all_neg_embeddings_ = src_global_neg_embeddings_;
        dst_all_neg_embeddings_ = dst_global_neg_embeddings_;
        return;
    }

    int num_per_chunk = (int) ceil((float) batch_size_ / num_chunks);

    // Sample for src and update filter
    int src_neg_size = src_global_neg_embeddings_.size(1);
    Indices src_sample = torch::randint(0, batch_size_, {num_chunks, num_local_negatives}, src_global_neg_embeddings_.device()).to(torch::kInt64);
    auto chunk_ids = (src_sample.div(num_per_chunk, "trunc")).view({num_chunks, -1});
    auto inv_mask = chunk_ids - torch::arange(0, num_chunks, src_global_neg_embeddings_.device()).view({num_chunks, -1});
    auto mask = (inv_mask == 0);
    auto temp_idx = torch::nonzero(mask);
    auto id_offset = src_sample.flatten(0, 1).index_select(0, (temp_idx.select(1, 0) * num_local_negatives + temp_idx.select(1, 1)));
    auto sample_offset = temp_idx.select(1, 1);
    src_all_neg_embeddings_ = torch::cat({src_global_neg_embeddings_, src_pos_embeddings_.index_select(0, src_sample.flatten(0, 1)).view({num_chunks, num_local_negatives, -1})}, 1);
    src_neg_filter_ = id_offset * src_all_neg_embeddings_.size(1) + (src_neg_size + sample_offset);

    // Sample for dst and update filter
    int dst_neg_size = dst_global_neg_embeddings_.size(1);
    Indices dst_sample = torch::randint(0, batch_size_, {num_chunks, num_local_negatives}, dst_global_neg_embeddings_.device()).to(torch::kInt64);
    chunk_ids = (dst_sample.div(num_per_chunk, "trunc")).view({num_chunks, -1});
    inv_mask = chunk_ids - torch::arange(0, num_chunks, dst_global_neg_embeddings_.device()).view({num_chunks, -1});
    mask = (inv_mask == 0);
    temp_idx = torch::nonzero(mask);
    id_offset = dst_sample.flatten(0, 1).index_select(0, (temp_idx.select(1, 0) * num_local_negatives + temp_idx.select(1, 1)));
    sample_offset = temp_idx.select(1, 1);
    dst_all_neg_embeddings_ = torch::cat({dst_global_neg_embeddings_, dst_pos_embeddings_.index_select(0, dst_sample.flatten(0, 1)).view({num_chunks, num_local_negatives, -1})}, 1);
    dst_neg_filter_ = id_offset * dst_all_neg_embeddings_.size(1) + (dst_neg_size + sample_offset);
}

void Batch::setUniqueNodes(bool use_neighbors, bool set_mapping) {
    if (use_neighbors) {
        unique_node_indices_ = gnn_graph_.getNodeIDs();

        auto tup = torch::sort(unique_node_indices_);
        Indices sorted_node_ids = std::get<0>(tup);
        Indices node_id_mapping = std::get<1>(tup);

        src_pos_indices_mapping_ = node_id_mapping.index_select(0, torch::searchsorted(sorted_node_ids, src_pos_indices_.to(sorted_node_ids.device()).contiguous()));
        dst_pos_indices_mapping_ = node_id_mapping.index_select(0, torch::searchsorted(sorted_node_ids, dst_pos_indices_.to(sorted_node_ids.device()).contiguous()));
        src_neg_indices_mapping_ = node_id_mapping.index_select(0, torch::searchsorted(sorted_node_ids, src_neg_indices_.to(sorted_node_ids.device()).contiguous()));
        dst_neg_indices_mapping_ = node_id_mapping.index_select(0, torch::searchsorted(sorted_node_ids, dst_neg_indices_.to(sorted_node_ids.device()).contiguous()));

    } else {
        Indices emb_idx;

        auto device = src_pos_indices_.device();
        emb_idx = torch::cat({src_pos_indices_.to(device), dst_pos_indices_.to(device), src_neg_indices_.to(device), dst_neg_indices_.to(device)});

        auto unique_tup = torch::_unique2(emb_idx, true, set_mapping, false);
        unique_node_indices_ = get<0>(unique_tup);

        if (set_mapping) {
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
        }
    }
}

void Batch::to(torch::Device device, at::cuda::CUDAStream *compute_stream) {

    at::cuda::CUDAStream transfer_stream = at::cuda::getStreamFromPool(false, device.index());
    at::cuda::CUDAStreamGuard stream_guard(transfer_stream);

    if (device.is_cuda()) {
        host_transfer_ = CudaEvent(device.index());
    }

    rel_indices_ = transfer_tensor(rel_indices_, device, compute_stream, &transfer_stream);

    root_node_indices_ = transfer_tensor(root_node_indices_, device, compute_stream, &transfer_stream);

    unique_node_indices_ = transfer_tensor(unique_node_indices_, device, compute_stream, &transfer_stream);

    unique_node_labels_ = transfer_tensor(unique_node_labels_, device, compute_stream, &transfer_stream);

    src_pos_indices_mapping_ = transfer_tensor(src_pos_indices_mapping_, device, compute_stream, &transfer_stream);
    dst_pos_indices_mapping_ = transfer_tensor(dst_pos_indices_mapping_, device, compute_stream, &transfer_stream);
    src_neg_indices_mapping_ = transfer_tensor(src_neg_indices_mapping_, device, compute_stream, &transfer_stream);
    dst_neg_indices_mapping_ = transfer_tensor(dst_neg_indices_mapping_, device, compute_stream, &transfer_stream);

    unique_node_embeddings_ = transfer_tensor(unique_node_embeddings_, device, compute_stream, &transfer_stream);
    if (train_) {
        unique_node_embeddings_state_ = transfer_tensor(unique_node_embeddings_state_, device, compute_stream, &transfer_stream);
    }

    unique_node_features_ = transfer_tensor(unique_node_features_, device, compute_stream, &transfer_stream);

    encoded_uniques_ = transfer_tensor(encoded_uniques_, device, compute_stream, &transfer_stream);

    if (gnn_graph_.node_ids_.defined()) {
        gnn_graph_.to(device, compute_stream, &transfer_stream);
    }

//    transfer_stream.synchronize();

    status_ = BatchStatus::TransferredToDevice;
}

void Batch::prepareBatch() {

    int num_chunks = negative_sampling_->num_chunks;
    int num_negatives = negative_sampling_->negatives_per_positive * (1 - negative_sampling_->degree_fraction);

    if (encoded_uniques_.defined()) {
        int64_t num_unencoded = unique_node_embeddings_.size(0) - encoded_uniques_.size(0);
        src_pos_embeddings_ = encoded_uniques_.index_select(0, src_pos_indices_mapping_ - num_unencoded);
        dst_pos_embeddings_ = encoded_uniques_.index_select(0, dst_pos_indices_mapping_ - num_unencoded);
        src_global_neg_embeddings_ = encoded_uniques_.index_select(0, src_neg_indices_mapping_ - num_unencoded).reshape({num_chunks, num_negatives, encoded_uniques_.size(-1)});
        dst_global_neg_embeddings_ = encoded_uniques_.index_select(0, dst_neg_indices_mapping_ - num_unencoded).reshape({num_chunks, num_negatives, encoded_uniques_.size(-1)});
    } else {
        src_pos_embeddings_ = unique_node_embeddings_.index_select(0, src_pos_indices_mapping_);
        dst_pos_embeddings_ = unique_node_embeddings_.index_select(0, dst_pos_indices_mapping_);
        src_global_neg_embeddings_ = unique_node_embeddings_.index_select(0, src_neg_indices_mapping_).reshape({num_chunks, num_negatives, unique_node_embeddings_.size(-1)});
        dst_global_neg_embeddings_ = unique_node_embeddings_.index_select(0, dst_neg_indices_mapping_).reshape({num_chunks, num_negatives, unique_node_embeddings_.size(-1)});
    }

    SPDLOG_TRACE("Batch: {} prepared for compute", batch_id_);
    status_ = BatchStatus::PreparedForCompute;
}

void Batch::accumulateGradients(float learning_rate) {

    if (src_pos_embeddings_.defined()) {
        unique_node_gradients_ = unique_node_embeddings_.grad();
        SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);

        unique_node_state_update_ = unique_node_gradients_.pow(2);
        unique_node_embeddings_state_.add_(unique_node_state_update_);
        unique_node_gradients_ = -learning_rate * (unique_node_gradients_ / ((unique_node_embeddings_state_).sqrt().add_(1e-10)));

        SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);
    }

    src_pos_embeddings_ = torch::Tensor();
    dst_pos_embeddings_ = torch::Tensor();
    src_global_neg_embeddings_ = torch::Tensor();
    dst_global_neg_embeddings_ = torch::Tensor();
    src_all_neg_embeddings_ = torch::Tensor();
    dst_all_neg_embeddings_ = torch::Tensor();

    unique_node_embeddings_state_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;
}

void Batch::embeddingsToHost() {

    if (unique_node_gradients_.device().type() == torch::kCUDA) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(unique_node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(unique_node_gradients_, true);
        Gradients temp_grads2 = torch::empty(unique_node_state_update_.sizes(), grad_opts);
        temp_grads2.copy_(unique_node_state_update_, true);
        unique_node_gradients_ = temp_grads;
        unique_node_state_update_ = temp_grads2;
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(torch::kCPU);
    }

    host_transfer_.record();
    host_transfer_.synchronize();
    status_ = BatchStatus::TransferredToHost;
}

void Batch::clear() {

    root_node_indices_ = torch::Tensor();
    unique_node_indices_ = torch::Tensor();
    unique_node_embeddings_ = torch::Tensor();
    unique_node_gradients_ = torch::Tensor();
    unique_node_state_update_ = torch::Tensor();
    unique_node_embeddings_state_ = torch::Tensor();

    unique_node_features_ = torch::Tensor();
    unique_node_labels_ = torch::Tensor();

    src_pos_indices_mapping_ = torch::Tensor();
    dst_pos_indices_mapping_ = torch::Tensor();
    src_neg_indices_mapping_ = torch::Tensor();
    dst_neg_indices_mapping_ = torch::Tensor();

    src_pos_indices_ = torch::Tensor();
    dst_pos_indices_ = torch::Tensor();
    rel_indices_ = torch::Tensor();
    src_neg_indices_ = torch::Tensor();
    dst_neg_indices_ = torch::Tensor();

    gnn_graph_.clear();
    encoded_uniques_ = torch::Tensor();

    src_pos_embeddings_ = torch::Tensor();
    dst_pos_embeddings_ = torch::Tensor();
    src_global_neg_embeddings_ = torch::Tensor();
    dst_global_neg_embeddings_ = torch::Tensor();
    src_all_neg_embeddings_ = torch::Tensor();
    dst_all_neg_embeddings_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();

    src_neg_filter_eval_ = {};
    dst_neg_filter_eval_ = {};

    SPDLOG_TRACE("Batch: {} cleared", batch_id_);
    status_ = BatchStatus::Done;
}