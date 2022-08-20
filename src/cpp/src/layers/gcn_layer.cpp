//
// Created by Jason Mohoney on 9/29/21.
//

#include "layers/gcn_layer.h"
#include "layers/layer_helpers.h"

GCNLayer::GCNLayer(shared_ptr<GNNLayerConfig> layer_config, bool use_incoming, bool use_outgoing, torch::Device device) : device_(torch::Device(torch::kCPU)) {
    layer_config_ = layer_config;
    options_ = layer_config->options;
    input_dim_ = options_->output_dim;
    output_dim_ = options_->input_dim;
    use_incoming_ = use_incoming;
    use_outgoing_ = use_outgoing;
    device_ = device;

    reset();
}

void GCNLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    torch::Tensor edge_mat = initialize_tensor(layer_config_->init,
                                               {output_dim_, input_dim_},
                                               tensor_options).set_requires_grad(true);

    w_ = register_parameter("w", edge_mat);
    if (layer_config_->bias) {
        torch::Tensor bias = initialize_tensor(layer_config_->bias_init,
                                               {output_dim_},
                                               tensor_options).set_requires_grad(true);

        bias_ = register_parameter("bias", bias);
    }
}

Embeddings GCNLayer::forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {

    torch::Tensor total_num_neighbors;
    torch::Tensor a_i;

    if (use_outgoing_) {
        Indices outgoing_neighbors = gnn_graph.getNeighborIDs(false, false);
        Indices outgoing_neighbor_offsets = gnn_graph.getNeighborOffsets(false);
        torch::Tensor outgoing_num = gnn_graph.getNumNeighbors(false);
        total_num_neighbors = outgoing_num;

        Embeddings outgoing_embeddings = inputs.index_select(0, outgoing_neighbors);
        torch::Tensor outgoing_normalization = torch::sqrt(gnn_graph.node_properties_.index_select(0, outgoing_neighbors) + 1).unsqueeze(-1);
        outgoing_embeddings = outgoing_embeddings / outgoing_normalization;
        a_i = segmented_sum_with_offsets(outgoing_embeddings, outgoing_neighbor_offsets);
    }


    if (use_incoming_) {
        Indices incoming_neighbors = gnn_graph.getNeighborIDs(true, false);
        Indices incoming_neighbor_offsets = gnn_graph.getNeighborOffsets(true);
        torch::Tensor incoming_num = gnn_graph.getNumNeighbors(true);

        if (total_num_neighbors.defined()) {
            total_num_neighbors = total_num_neighbors + incoming_num;
        } else {
            total_num_neighbors = incoming_num;
        }

        Embeddings incoming_embeddings = inputs.index_select(0, incoming_neighbors);
        torch::Tensor incoming_normalization = torch::sqrt(gnn_graph.node_properties_.index_select(0, incoming_neighbors) + 1).unsqueeze(-1);
        incoming_embeddings = incoming_embeddings / incoming_normalization;

        if (a_i.defined()) {
            a_i = a_i + segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        } else {
            a_i = segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        }
    }

    int64_t layer_offset = gnn_graph.getLayerOffset();
    Embeddings self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);

    a_i = a_i + self_embs / torch::sqrt((total_num_neighbors + 1)).unsqueeze(-1);
    a_i = a_i / torch::sqrt((total_num_neighbors + 1)).unsqueeze(-1);
    Embeddings outputs = torch::matmul(w_, a_i.transpose(0, -1)).transpose(0, -1);
    return outputs + bias_;
}