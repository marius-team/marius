//
// Created by Jason Mohoney on 12/10/21.
//

#include "nn/layers/reduction/linear.h"

#include "nn/initialization.h"

LinearReduction::LinearReduction(shared_ptr<LayerConfig> layer_config, torch::Device device) {
    config_ = layer_config;
    device_ = device;
    reset();
}

torch::Tensor LinearReduction::forward(std::vector<torch::Tensor> inputs) {
    torch::Tensor tmp = torch::cat(inputs, 1).transpose(0, -1);
    torch::Tensor outputs = torch::matmul(weight_matrix_, tmp);
    return outputs.transpose(0, -1);
}

void LinearReduction::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    torch::Tensor weight_mat = initialize_tensor(config_->init, {config_->output_dim, config_->input_dim}, tensor_options).set_requires_grad(true);

    weight_matrix_ = register_parameter("weight_matrix", weight_mat);

    if (config_->bias) {
        init_bias();
    }
}