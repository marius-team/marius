//
// Created by Jason Mohoney on 9/29/21.
//

#include "layers/gat_layer.h"

#include "layers/layer_helpers.h"

GATLayer::GATLayer(shared_ptr<GNNLayerConfig> layer_config, torch::Device device) : device_(torch::Device(torch::kCPU)) {

    layer_config_ = layer_config;
    options_ = std::dynamic_pointer_cast<GATLayerOptions>(layer_config->options);
    input_dim_ = layer_config->options->input_dim;
    output_dim_ = layer_config->options->output_dim;

    device_ = device;

    input_dropout_ = options_->input_dropout;
    attention_dropout_ = options_->attention_dropout;

    if (options_->average_heads) {
        head_dim_ = output_dim_;
    } else {
        assert(output_dim_ % options_->num_heads == 0);
        head_dim_ = output_dim_ / options_->num_heads;
    }

    reset();
}

void GATLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    torch::Tensor weight_matrices = initialize_tensor(layer_config_->init,
                                                      {head_dim_ * options_->num_heads, input_dim_},
                                                      tensor_options,
                                                      {input_dim_, head_dim_}).set_requires_grad(true);

    torch::Tensor a_l = initialize_tensor(layer_config_->init,
                                          {options_->num_heads, 1, head_dim_},
                                          tensor_options,
                                          {head_dim_, 1}).set_requires_grad(true);

    torch::Tensor a_r = initialize_tensor(layer_config_->init,
                                          {options_->num_heads, 1, head_dim_},
                                          tensor_options,
                                          {head_dim_, 1}).set_requires_grad(true);

    weight_matrices_ = register_parameter("weight_matrices", weight_matrices);
    a_l_ = register_parameter("a_l", a_l);
    a_r_ = register_parameter("a_r", a_r);

    if (layer_config_->bias) {
        torch::Tensor bias = initialize_tensor(layer_config_->bias_init,
                                               {1, output_dim_},
                                               tensor_options).set_requires_grad(true);
        bias_ = register_parameter("bias", bias);
    }

}

Embeddings GATLayer::forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {
    auto relu_opts = torch::nn::LeakyReLUOptions();
    relu_opts.negative_slope(options_->negative_slope);
    auto leaky_relu = torch::nn::LeakyReLU(relu_opts);

    Indices incoming_neighbors = gnn_graph.getNeighborIDs(true, false);
    Indices incoming_neighbor_offsets = gnn_graph.getNeighborOffsets(true);

    torch::Tensor incoming_total_neighbors = gnn_graph.getNumNeighbors(true);

    int64_t layer_offset = gnn_graph.getLayerOffset();
    torch::Tensor parent_ids = torch::arange(inputs.size(0) - layer_offset, incoming_total_neighbors.options())
            .repeat_interleave(incoming_total_neighbors);

    if (train && input_dropout_ > 0) {
        auto dropout_opts = torch::nn::DropoutOptions().p(input_dropout_).inplace(false);
        auto dropout = torch::nn::Dropout(dropout_opts);
        inputs = dropout(inputs);
    }

    Embeddings incoming_embeddings = inputs.index_select(0, incoming_neighbors);
    Embeddings incoming_transforms = torch::matmul(weight_matrices_, incoming_embeddings.transpose(0, 1));
    incoming_transforms = incoming_transforms.reshape({options_->num_heads, head_dim_, -1});

    // free memory as this tensor can become large with large numbers of neighbors
    incoming_embeddings = torch::Tensor();

    Embeddings self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);
    Embeddings self_transforms = torch::matmul(weight_matrices_, self_embs.transpose(0, 1));
    self_transforms = self_transforms.reshape({options_->num_heads, head_dim_, -1});
    torch::Tensor self_transforms_l = torch::matmul(a_l_, self_transforms);

    torch::Tensor self_atn_weights = self_transforms_l + torch::matmul(a_r_, self_transforms);
    self_atn_weights = leaky_relu(self_atn_weights);

    self_transforms_l = self_transforms_l.index_select(-1, parent_ids);
    torch::Tensor nbr_atn_weights = self_transforms_l + torch::matmul(a_r_, incoming_transforms);
    nbr_atn_weights = leaky_relu(nbr_atn_weights);

    nbr_atn_weights = nbr_atn_weights.transpose(0, 2);   // [total_num_nbrs, 1, num_heads_]
    self_atn_weights = self_atn_weights.transpose(0, 2); // [num_to_encode, 1, num_heads_]

    std::tie(nbr_atn_weights, self_atn_weights) = attention_softmax(nbr_atn_weights, self_atn_weights,
                                                                    incoming_neighbor_offsets, parent_ids,
                                                                    incoming_total_neighbors);

    nbr_atn_weights = nbr_atn_weights.transpose(0, 2);
    self_atn_weights = self_atn_weights.transpose(0, 2);

    if (train && attention_dropout_ > 0) {
        auto dropout_opts = torch::nn::DropoutOptions().p(attention_dropout_).inplace(false);
        auto dropout = torch::nn::Dropout(dropout_opts);

        nbr_atn_weights = dropout(nbr_atn_weights);
        self_atn_weights = dropout(self_atn_weights);
    }

    nbr_atn_weights = nbr_atn_weights.repeat({1, head_dim_, 1});

    torch::Tensor tmp = (incoming_transforms * nbr_atn_weights).transpose(0, 2);
    torch::Tensor h_i = segmented_sum(tmp, parent_ids, incoming_total_neighbors.size(0));
    h_i = h_i.transpose(0, 2);

    tmp = self_transforms * self_atn_weights;
    h_i = h_i + tmp;

    if (options_->average_heads) {
        h_i = torch::mean(h_i, 0);
    } else {
        h_i = h_i.reshape({output_dim_, -1});
    }

    h_i = h_i.transpose(0, 1);

    if (layer_config_->bias) {
        h_i = h_i + bias_;
    }

    return h_i;
}