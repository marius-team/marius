//
// Created by Jason Mohoney on 12/10/21.
//

#include "nn/layers/reduction/concat.h"

ConcatReduction::ConcatReduction(shared_ptr<LayerConfig> layer_config, torch::Device device) {
    config_ = layer_config;
    device_ = device;
}

torch::Tensor ConcatReduction::forward(std::vector<torch::Tensor> inputs) { return torch::cat(inputs, 1); }

void ConcatReduction::reset() {
    if (config_->bias) {
        init_bias();
    }
}