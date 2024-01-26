//
// Created by Jason Mohoney on 7/9/20.
//

#include "data/batch.h"

#include "configuration/constants.h"
#include "reporting/logger.h"

using std::get;

Batch::Batch(bool train) : device_transfer_(0), host_transfer_(0), timer_(false) {
    status_ = BatchStatus::Waiting;
    train_ = train;
    device_id_ = -1;
    clear();
}

Batch::~Batch() { clear(); }

void Batch::to(torch::Device device, CudaStream *compute_stream) {
    CudaStream transfer_stream = getStreamFromPool(false, device.index());
    CudaStreamGuard stream_guard(transfer_stream);

    if (device.is_cuda()) {
        host_transfer_ = CudaEvent(device.index());
    }

    edges_ = transfer_tensor(edges_, device, compute_stream, &transfer_stream);

    edges_weights_ = transfer_tensor(edges_, device, compute_stream, &transfer_stream);

    neg_edges_ = transfer_tensor(neg_edges_, device, compute_stream, &transfer_stream);

    root_node_indices_ = transfer_tensor(root_node_indices_, device, compute_stream, &transfer_stream);

    unique_node_indices_ = transfer_tensor(unique_node_indices_, device, compute_stream, &transfer_stream);

    node_labels_ = transfer_tensor(node_labels_, device, compute_stream, &transfer_stream);

    src_neg_indices_mapping_ = transfer_tensor(src_neg_indices_mapping_, device, compute_stream, &transfer_stream);

    dst_neg_indices_mapping_ = transfer_tensor(dst_neg_indices_mapping_, device, compute_stream, &transfer_stream);

    src_neg_filter_ = transfer_tensor(src_neg_filter_, device, compute_stream, &transfer_stream);

    dst_neg_filter_ = transfer_tensor(dst_neg_filter_, device, compute_stream, &transfer_stream);

    node_embeddings_ = transfer_tensor(node_embeddings_, device, compute_stream, &transfer_stream);

    node_embeddings_state_ = transfer_tensor(node_embeddings_state_, device, compute_stream, &transfer_stream);

    node_features_ = transfer_tensor(node_features_, device, compute_stream, &transfer_stream);

    encoded_uniques_ = transfer_tensor(encoded_uniques_, device, compute_stream, &transfer_stream);

    if (dense_graph_.node_ids_.defined()) {
        dense_graph_.to(device, compute_stream, &transfer_stream);
    }

    status_ = BatchStatus::TransferredToDevice;
}

void Batch::accumulateGradients(float learning_rate) {
    if (node_embeddings_.defined()) {
        node_gradients_ = node_embeddings_.grad();
        SPDLOG_TRACE("Batch: {} accumulated node gradients", batch_id_);

        node_state_update_ = node_gradients_.pow(2);
        node_embeddings_state_.add_(node_state_update_);
        node_gradients_ = -learning_rate * (node_gradients_ / (node_embeddings_state_.sqrt().add_(1e-10)));

        SPDLOG_TRACE("Batch: {} adjusted gradients", batch_id_);
    }

    node_embeddings_state_ = torch::Tensor();

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;
}

void Batch::embeddingsToHost() {
    if (node_gradients_.defined() && node_gradients_.device().is_cuda()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(node_gradients_, true);
        Gradients temp_grads2 = torch::empty(node_state_update_.sizes(), grad_opts);
        temp_grads2.copy_(node_state_update_, true);
        node_gradients_ = temp_grads;
        node_state_update_ = temp_grads2;
    }

    if (unique_node_indices_.defined()) {
        unique_node_indices_ = unique_node_indices_.to(torch::kCPU);
    }

    if (encoded_uniques_.defined()) {
        encoded_uniques_ = encoded_uniques_.to(torch::kCPU);
    }

    host_transfer_.record();
    host_transfer_.synchronize();
    status_ = BatchStatus::TransferredToHost;
}

void Batch::clear() {
    root_node_indices_ = torch::Tensor();
    unique_node_indices_ = torch::Tensor();
    node_embeddings_ = torch::Tensor();
    node_gradients_ = torch::Tensor();
    node_state_update_ = torch::Tensor();
    node_embeddings_state_ = torch::Tensor();

    node_features_ = torch::Tensor();
    node_labels_ = torch::Tensor();

    src_neg_indices_mapping_ = torch::Tensor();
    dst_neg_indices_mapping_ = torch::Tensor();

    edges_ = torch::Tensor();
    neg_edges_ = torch::Tensor();
    src_neg_indices_ = torch::Tensor();
    dst_neg_indices_ = torch::Tensor();

    dense_graph_.clear();
    encoded_uniques_ = torch::Tensor();

    src_neg_filter_ = torch::Tensor();
    dst_neg_filter_ = torch::Tensor();
}