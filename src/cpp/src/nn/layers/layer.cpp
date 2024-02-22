//
// Created by Jason Mohoney on 2/1/22.
//

#include "nn/layers/layer.h"

Layer::Layer() : device_(torch::kCPU) {}

torch::Tensor Layer::post_hook(torch::Tensor input) {
    if (config_->bias) {
        input = input + bias_;
    }
    input = apply_activation(config_->activation, input);

    return input;
}

void Layer::init_bias() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    torch::Tensor bias = initialize_tensor(config_->bias_init, {config_->output_dim}, tensor_options).set_requires_grad(true);
    bias_ = register_parameter("bias", bias);
}