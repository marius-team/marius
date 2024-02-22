//
// Created by Jason Mohoney on 2/1/22.
//

#include "nn/layers/feature/feature.h"

FeatureLayer::FeatureLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int offset) {
    config_ = layer_config;
    offset_ = offset;
    device_ = device;

    reset();
}

torch::Tensor FeatureLayer::forward(torch::Tensor input) { return input.narrow(1, offset_, config_->output_dim); }

void FeatureLayer::reset() {
    if (config_->bias) {
        init_bias();
    }
}