//
// Created by Jason Mohoney on 9/29/21.
//

#include "layers/graph_sage_layer.h"

#include "layers/layer_helpers.h"
#include "logger.h"

GraphSageLayer::GraphSageLayer(shared_ptr<GNNLayerConfig> layer_config, bool use_incoming, bool use_outgoing, torch::Device device) : device_(torch::Device(torch::kCPU)) {
    layer_config_ = layer_config;
    options_ = std::dynamic_pointer_cast<GraphSageLayerOptions>(layer_config_->options);
    use_incoming_ = use_incoming;
    use_outgoing_ = use_outgoing;
    input_dim_ = options_->input_dim;
    output_dim_ = options_->output_dim;
    device_ = device;

    reset();
}

void GraphSageLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    // Note: need to multiply the fans by 1/2 to match DGL's initialization
    torch::Tensor edge_mat = initialize_tensor(layer_config_->init,
                                               {output_dim_, input_dim_},
                                               tensor_options).set_requires_grad(true);
//    torch::Tensor edge_mat = initialize_tensor(layer_config_->init,
//                                               {output_dim_, input_dim_},
//                                               tensor_options,
//                                               {(int64_t)(0.5*output_dim_), (int64_t)(0.5*input_dim_)}).set_requires_grad(true);
    w1_ = register_parameter("w1", edge_mat);

    if (options_->aggregator == GraphSageAggregator::MEAN) {
        edge_mat = initialize_tensor(layer_config_->init,
                                     {output_dim_, input_dim_},
                                     tensor_options).set_requires_grad(true);
//        edge_mat = initialize_tensor(layer_config_->init,
//                                     {output_dim_, input_dim_},
//                                     tensor_options,
//                                     {(int64_t)(0.5*output_dim_), (int64_t)(0.5*input_dim_)}).set_requires_grad(true);
        w2_ = register_parameter("w2", edge_mat);
    }


    if (layer_config_->bias) {
        torch::Tensor bias = initialize_tensor(layer_config_->bias_init,
                                               {output_dim_},
                                               tensor_options).set_requires_grad(true);
        bias_ = register_parameter("bias", bias);
    }
}

Embeddings GraphSageLayer::forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {

    torch::Tensor total_num_neighbors;
    torch::Tensor a_i;

    if (use_outgoing_) {
        Indices outgoing_neighbors = gnn_graph.getNeighborIDs(false, false);
        Indices outgoing_neighbor_offsets = gnn_graph.getNeighborOffsets(false);
        torch::Tensor outgoing_num = gnn_graph.getNumNeighbors(false);

        Embeddings outgoing_embeddings = inputs.index_select(0, outgoing_neighbors);

        total_num_neighbors = outgoing_num;
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

        if (a_i.defined()) {
            a_i = a_i + segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        } else {
            a_i = segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        }
    }

    int64_t layer_offset = gnn_graph.getLayerOffset();
    Embeddings self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);

    Embeddings outputs;
    if (options_->aggregator == GraphSageAggregator::GCN) {
        a_i = a_i + self_embs;
        a_i = a_i / (total_num_neighbors + 1).unsqueeze(-1);
        outputs = torch::matmul(w1_, a_i.transpose(0, -1)).transpose(0, -1);
    } else if (options_->aggregator == GraphSageAggregator::MEAN) {
        torch::Tensor denominator = torch::where(torch::not_equal(total_num_neighbors, 0), total_num_neighbors, 1).to(a_i.dtype()).unsqueeze(-1);
        a_i = a_i / denominator;
        outputs = (torch::matmul(w1_, self_embs.transpose(0, -1)) + torch::matmul(w2_, a_i.transpose(0, -1))).transpose(0, -1);
    } else {
        throw std::runtime_error("Unrecognized aggregator");
    }

    return outputs + bias_;
}