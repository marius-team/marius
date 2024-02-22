//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/layers/gnn/rgcn_layer.h"

#include "nn/layers/gnn/layer_helpers.h"

RGCNLayer::RGCNLayer(shared_ptr<LayerConfig> layer_config, int num_relations, torch::Device device) {
    config_ = layer_config;
    options_ = std::dynamic_pointer_cast<GNNLayerOptions>(config_->options);
    num_relations_ = num_relations;
    input_dim_ = layer_config->input_dim;
    output_dim_ = layer_config->output_dim;
    device_ = device;

    reset();
}

void RGCNLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    torch::Tensor rel_mats = initialize_tensor(config_->init, {num_relations_, output_dim_, input_dim_}, tensor_options).set_requires_grad(true);
    relation_matrices_ = register_parameter("relation_matrices", rel_mats);

    //    if (use_incoming_) {
    //        torch::Tensor inverse_rel_mats = initialize_tensor(config_->init,
    //                                                          {num_relations_, output_dim_, input_dim_},
    //                                                           tensor_options).set_requires_grad(true);
    //        inverse_relation_matrices_ = register_parameter("inverse_relation_matrices_", inverse_rel_mats);
    //    }

    torch::Tensor self_mat = initialize_tensor(config_->init, {output_dim_, input_dim_}, tensor_options).set_requires_grad(true);
    self_matrix_ = register_parameter("self_matrix", self_mat);

    if (config_->bias) {
        init_bias();
    }
}

torch::Tensor RGCNLayer::forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train) {
    Indices outgoing_neighbors = dense_graph.getNeighborIDs(false, false);
    Indices outgoing_relations = dense_graph.getRelationIDs(false);
    Indices outgoing_neighbor_offsets = dense_graph.getNeighborOffsets(false);
    torch::Tensor outgoing_num = dense_graph.getNumNeighbors(false);
    torch::Tensor total_num_neighbors = outgoing_num;

    if (!outgoing_relations.defined()) {
        outgoing_relations = torch::zeros_like(outgoing_neighbors);
    }

    torch::Tensor outgoing_embeddings = inputs.index_select(0, outgoing_neighbors);
    torch::Tensor outgoing_relation_matrices = relation_matrices_.index_select(0, outgoing_relations);
    outgoing_embeddings = torch::bmm(outgoing_relation_matrices, outgoing_embeddings.unsqueeze(2)).flatten(1, 2);

    torch::Tensor a_i = segmented_sum_with_offsets(outgoing_embeddings, outgoing_neighbor_offsets);

    //    if (dense_graph.in_neighbors_mapping_.defined()) {
    //        Indices incoming_neighbors = dense_graph.getNeighborIDs(true, false);
    //        Indices incoming_relations = dense_graph.getRelationIDs(true);
    //        Indices incoming_neighbor_offsets = dense_graph.getNeighborOffsets(true);
    //        torch::Tensor incoming_num = dense_graph.getNumNeighbors(true);
    //        total_num_neighbors = total_num_neighbors + incoming_num;
    //
    //        if (!incoming_relations.defined()) {
    //            incoming_relations = torch::zeros_like(incoming_neighbors);
    //        }
    //
    //        torch::Tensor incoming_embeddings = inputs.index_select(0, incoming_neighbors);
    //        torch::Tensor incoming_relation_matrices = inverse_relation_matrices_.index_select(0, incoming_relations);
    //        incoming_embeddings = torch::bmm(incoming_relation_matrices, incoming_embeddings.unsqueeze(2)).flatten(1, 2);
    //
    //        a_i = a_i + segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
    //    }
    torch::Tensor denominator = torch::where(torch::not_equal(total_num_neighbors, 0), total_num_neighbors, 1).to(a_i.dtype()).unsqueeze(-1);
    a_i = a_i / denominator;

    int64_t layer_offset = dense_graph.getLayerOffset();
    torch::Tensor self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);

    // clone might be needed for async parameter updates
    self_embs = torch::matmul(self_matrix_, self_embs.transpose(0, -1)).transpose(0, -1);

    // clone might be needed for async parameter updates
    torch::Tensor outputs = a_i + self_embs + bias_;

    return outputs;
}