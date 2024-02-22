//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/layers/gnn/gcn_layer.h"

#include "nn/layers/gnn/layer_helpers.h"

GCNLayer::GCNLayer(shared_ptr<LayerConfig> layer_config, torch::Device device) {
    config_ = layer_config;
    options_ = std::dynamic_pointer_cast<GNNLayerOptions>(config_->options);
    input_dim_ = config_->output_dim;
    output_dim_ = config_->input_dim;
    device_ = device;

    reset();
}

void GCNLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    torch::Tensor edge_mat = initialize_tensor(config_->init, {output_dim_, input_dim_}, tensor_options).set_requires_grad(true);

    w_ = register_parameter("w", edge_mat);
    if (config_->bias) {
        init_bias();
    }
}

torch::Tensor GCNLayer::forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train) {
    torch::Tensor total_num_neighbors;
    torch::Tensor a_i;

    if (dense_graph.out_neighbors_mapping_.defined()) {
        Indices outgoing_neighbors = dense_graph.getNeighborIDs(false, false);
        Indices outgoing_neighbor_offsets = dense_graph.getNeighborOffsets(false);
        torch::Tensor outgoing_num = dense_graph.getNumNeighbors(false);
        total_num_neighbors = outgoing_num;

        torch::Tensor outgoing_embeddings = inputs.index_select(0, outgoing_neighbors);
        torch::Tensor outgoing_normalization = torch::sqrt(dense_graph.node_properties_.index_select(0, outgoing_neighbors) + 1).unsqueeze(-1);
        outgoing_embeddings = outgoing_embeddings / outgoing_normalization;
        a_i = segmented_sum_with_offsets(outgoing_embeddings, outgoing_neighbor_offsets);
    }

    if (dense_graph.in_neighbors_mapping_.defined()) {
        Indices incoming_neighbors = dense_graph.getNeighborIDs(true, false);
        Indices incoming_neighbor_offsets = dense_graph.getNeighborOffsets(true);
        torch::Tensor incoming_num = dense_graph.getNumNeighbors(true);

        if (total_num_neighbors.defined()) {
            total_num_neighbors = total_num_neighbors + incoming_num;
        } else {
            total_num_neighbors = incoming_num;
        }

        torch::Tensor incoming_embeddings = inputs.index_select(0, incoming_neighbors);
        torch::Tensor incoming_normalization = torch::sqrt(dense_graph.node_properties_.index_select(0, incoming_neighbors) + 1).unsqueeze(-1);
        incoming_embeddings = incoming_embeddings / incoming_normalization;

        if (a_i.defined()) {
            a_i = a_i + segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        } else {
            a_i = segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        }
    }

    int64_t layer_offset = dense_graph.getLayerOffset();
    torch::Tensor self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);

    a_i = a_i + self_embs / torch::sqrt((total_num_neighbors + 1)).unsqueeze(-1);
    a_i = a_i / torch::sqrt((total_num_neighbors + 1)).unsqueeze(-1);
    torch::Tensor outputs = torch::matmul(w_, a_i.transpose(0, -1)).transpose(0, -1);
    return outputs + bias_;
}