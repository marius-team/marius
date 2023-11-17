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
    creator_id_ = -1;
    loss_ = 0;
    clear();
}

Batch::~Batch() { clear(); }

void Batch::to(torch::Device device, CudaStream *compute_stream) {

    if (sub_batches_.size() > 0) {
//        std::vector<torch::Device> tmp = {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)};

        std::vector<torch::Device> tmp;
        for (int i = 0; i < sub_batches_.size(); i++) {
            tmp.emplace_back(torch::Device(torch::kCUDA, i));
        }

        #pragma omp parallel for
        for (int i = 0; i < sub_batches_.size(); i++) {
            sub_batches_[i]->to(tmp[i]);
        }
        return;
    }

    CudaStream transfer_stream = getStreamFromPool(false, device.index());
    CudaStreamGuard stream_guard(transfer_stream);

//    if (device.is_cuda()) {
//        host_transfer_ = CudaEvent(device.index());
//    }

    edges_ = transfer_tensor(edges_, device, compute_stream, &transfer_stream);

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

void Batch::remoteTo(shared_ptr<c10d::ProcessGroupGloo> pg, int worker_id, int tag, bool send_meta) {

    if (send_meta) {
        torch::Tensor metadata = torch::tensor({batch_id_, batch_size_, creator_id_}, {torch::kInt32});

        std::vector<torch::Tensor> transfer_vec;
        transfer_vec.push_back(metadata);
        auto work = pg->send(transfer_vec, worker_id, tag);
        if (!work->wait()) {
            throw work->exception();
        }
    }

    if (sub_batches_.size() > 0) {
//        #pragma omp parallel for // TODO: need to look at whether this works or not (e.g., parallel sending)
        for (int i = 0; i < sub_batches_.size(); i++) {
            sub_batches_[i]->remoteTo(pg, worker_id, tag, false);
        }
        return;
    }

    send_tensor(edges_, pg, worker_id, tag);

    send_tensor(neg_edges_, pg, worker_id, tag);

    send_tensor(root_node_indices_, pg, worker_id, tag);

    send_tensor(unique_node_indices_, pg, worker_id, tag);

    send_tensor(node_labels_, pg, worker_id, tag);

    send_tensor(src_neg_indices_mapping_, pg, worker_id, tag);

    send_tensor(dst_neg_indices_mapping_, pg, worker_id, tag);

    send_tensor(src_neg_filter_, pg, worker_id, tag);

    send_tensor(dst_neg_filter_, pg, worker_id, tag);

    send_tensor(node_embeddings_, pg, worker_id, tag);

    send_tensor(node_embeddings_state_, pg, worker_id, tag);

    send_tensor(node_features_, pg, worker_id, tag);

    send_tensor(encoded_uniques_, pg, worker_id, tag);

    dense_graph_.send(pg, worker_id, tag);

    send_tensor(node_gradients_, pg, worker_id, tag);

    send_tensor(node_state_update_, pg, worker_id, tag);

    send_tensor(pos_scores_, pg, worker_id, tag);

    send_tensor(neg_scores_, pg, worker_id, tag);

    send_tensor(inv_pos_scores_, pg, worker_id, tag);

    send_tensor(inv_neg_scores_, pg, worker_id, tag);

    send_tensor(y_pred_, pg, worker_id, tag);

    // can clear batch, it's sent to another machine at this point
    clear();
}

void Batch::remoteReceive(shared_ptr<c10d::ProcessGroupGloo> pg, int worker_id, int tag, bool receive_meta) {

    if (receive_meta) {
        torch::Tensor metadata = torch::tensor({-1, -1, -1}, {torch::kInt32});

        std::vector <torch::Tensor> transfer_vec;
        transfer_vec.push_back(metadata);
        auto work = pg->recv(transfer_vec, worker_id, tag);
        if (!work->wait()) {
            throw work->exception();
        }

        batch_id_ = metadata[0].item<int>();
        batch_size_ = metadata[1].item<int>();
        creator_id_ = metadata[2].item<int>();
    }

    if (sub_batches_.size() > 0) {
//        #pragma omp parallel for // TODO: need to look at whether this works or not (e.g., parallel sending)
        for (int i = 0; i < sub_batches_.size(); i++) {
            sub_batches_[i]->remoteReceive(pg, worker_id, tag, false);
        }
        return;
    }

    edges_ = receive_tensor(pg, worker_id, tag);

    neg_edges_ = receive_tensor(pg, worker_id, tag);

    root_node_indices_ = receive_tensor(pg, worker_id, tag);

    unique_node_indices_ = receive_tensor(pg, worker_id, tag);

    node_labels_ = receive_tensor(pg, worker_id, tag);

    src_neg_indices_mapping_ = receive_tensor(pg, worker_id, tag);

    dst_neg_indices_mapping_ = receive_tensor(pg, worker_id, tag);

    src_neg_filter_ = receive_tensor(pg, worker_id, tag);

    dst_neg_filter_ = receive_tensor(pg, worker_id, tag);

    node_embeddings_ = receive_tensor(pg, worker_id, tag);

    node_embeddings_state_ = receive_tensor(pg, worker_id, tag);

    node_features_ = receive_tensor(pg, worker_id, tag);

    encoded_uniques_ = receive_tensor(pg, worker_id, tag);

    dense_graph_.receive(pg, worker_id, tag);

    node_gradients_ = receive_tensor(pg, worker_id, tag);

    node_state_update_ = receive_tensor(pg, worker_id, tag);

    pos_scores_ = receive_tensor(pg, worker_id, tag);

    neg_scores_ = receive_tensor(pg, worker_id, tag);

    inv_pos_scores_ = receive_tensor(pg, worker_id, tag);

    inv_neg_scores_ = receive_tensor(pg, worker_id, tag);

    y_pred_ = receive_tensor(pg, worker_id, tag);
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

    root_node_indices_ = torch::Tensor();
    node_embeddings_ = torch::Tensor();
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

    SPDLOG_TRACE("Batch: {} cleared gpu embeddings and gradients", batch_id_);

    status_ = BatchStatus::AccumulatedGradients;

    getCurrentCudaStream(node_gradients_.device().index()).synchronize();
}

void Batch::embeddingsToHost() {
    if (sub_batches_.size() > 0) {
        #pragma omp parallel for
        for (int i = 0; i < sub_batches_.size(); i++) {
            sub_batches_[i]->embeddingsToHost();
        }
        return;
    }

    if (node_gradients_.defined() && node_gradients_.device().is_cuda()) {
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
        Gradients temp_grads = torch::empty(node_gradients_.sizes(), grad_opts);
        temp_grads.copy_(node_gradients_, false);

        Gradients temp_grads2 = torch::empty(node_state_update_.sizes(), grad_opts);
        temp_grads2.copy_(node_state_update_, false);

        node_gradients_ = temp_grads;
        node_state_update_ = temp_grads2;
    }

    if (unique_node_indices_.defined()) {
        auto opts = torch::TensorOptions().dtype(unique_node_indices_.dtype()).device(torch::kCPU).pinned_memory(true);
        torch::Tensor temp = torch::empty(unique_node_indices_.sizes(), opts);
        temp.copy_(unique_node_indices_, false);

        unique_node_indices_ = temp;
    }

    if (encoded_uniques_.defined()) {
        auto opts = torch::TensorOptions().dtype(encoded_uniques_.dtype()).device(torch::kCPU).pinned_memory(true);
        torch::Tensor temp = torch::empty(encoded_uniques_.sizes(), opts);
        temp.copy_(encoded_uniques_, false);

        encoded_uniques_ = temp;
    }

//    host_transfer_.record();
//    host_transfer_.synchronize();
    status_ = BatchStatus::TransferredToHost;
}

void Batch::evalToHost() {
    if (sub_batches_.size() > 0) {
        throw MariusRuntimeException("Multi-GPU evaluation currently not supported.");
    }

    if (pos_scores_.defined()) {
        pos_scores_ = pos_scores_.to(torch::kCPU);
    }

    if (neg_scores_.defined()) {
        neg_scores_ = neg_scores_.to(torch::kCPU);
    }

    if (inv_pos_scores_.defined()) {
        inv_pos_scores_ = inv_pos_scores_.to(torch::kCPU);
    }

    if (inv_neg_scores_.defined()) {
        inv_neg_scores_ = inv_neg_scores_.to(torch::kCPU);
    }

    if (y_pred_.defined()) {
        y_pred_ = y_pred_.to(torch::kCPU);
    }

    if (node_labels_.defined()) {
        node_labels_ = node_labels_.to(torch::kCPU);
    }
}

void Batch::clear(bool clear_eval) {
    root_node_indices_ = torch::Tensor();
    unique_node_indices_ = torch::Tensor();
    node_embeddings_ = torch::Tensor();
    node_gradients_ = torch::Tensor();
    node_state_update_ = torch::Tensor();
    node_embeddings_state_ = torch::Tensor();

    node_features_ = torch::Tensor();
//    node_labels_ = torch::Tensor();

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

    if (sub_batches_.size() > 0) {
        #pragma omp parallel for
        for (int i = 0; i < sub_batches_.size(); i++) {
            sub_batches_[i]->clear();
        }
    }

    sub_batches_ = {};

    if (clear_eval) {
        pos_scores_ = torch::Tensor();
        neg_scores_ = torch::Tensor();
        inv_pos_scores_ = torch::Tensor();
        inv_neg_scores_ = torch::Tensor();
        y_pred_ = torch::Tensor();
        node_labels_ = torch::Tensor();
    }
}

double Batch::getLoss(LossReduction reduction_type) {
    double loss = 0.0;
    if (sub_batches_.size() > 0) {
        for (int i = 0; i < sub_batches_.size(); i++) {
            loss += sub_batches_[i]->loss_;
        }

        if (reduction_type == LossReduction::MEAN) {
            loss = loss / sub_batches_.size();
        }

    } else {
        loss = loss_;
    }

    return loss;
}