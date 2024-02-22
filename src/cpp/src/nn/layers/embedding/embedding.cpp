//
// Created by Jason Mohoney on 2/1/22.
//

#include "nn/layers/embedding/embedding.h"

#include "nn/initialization.h"

EmbeddingLayer::EmbeddingLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int offset) {
    config_ = layer_config;
    offset_ = offset;
    device_ = device;

    reset();
}

torch::Tensor EmbeddingLayer::forward(torch::Tensor input) { return input.narrow(1, offset_, config_->output_dim); }

torch::Tensor EmbeddingLayer::init_embeddings(int64_t num_nodes) {
    auto options = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    torch::Tensor embs = initialize_tensor(config_->init, {num_nodes, config_->output_dim}, options);

    return embs;
}

void EmbeddingLayer::reset() {
    if (config_->bias) {
        init_bias();
    }
}