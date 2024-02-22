//
// Created by Jason Mohoney on 10/7/21.
//

#include "nn/initialization.h"

std::tuple<int64_t, int64_t> compute_fans(std::vector<int64_t> shape) {
    int64_t fan_in = 0;
    int64_t fan_out = 0;

    if (shape.size() < 1) {
        fan_in = fan_out = 1;
    } else if (shape.size() == 1) {
        fan_in = fan_out = shape[0];
    } else if (shape.size() == 2) {
        fan_in = shape[0];
        fan_out = shape[1];
    } else {
        fan_in = shape[shape.size() - 2];
        fan_out = shape[shape.size() - 1];
    }

    return std::forward_as_tuple(fan_in, fan_out);
}

torch::Tensor glorot_uniform(std::vector<int64_t> shape, std::tuple<int64_t, int64_t> fans, torch::TensorOptions options) {
    int64_t fan_in = std::get<0>(fans);
    int64_t fan_out = std::get<1>(fans);

    if (fan_in == -1 || fan_out == -1) {
        auto tup = compute_fans(shape);
        fan_in = std::get<0>(tup);
        fan_out = std::get<1>(tup);
    }

    float limit = sqrt(6.0 / (fan_in + fan_out));
    torch::Tensor ret = torch::rand(shape, options);
    ret = 2 * limit * (ret - .5);

    return ret;
}

torch::Tensor glorot_normal(std::vector<int64_t> shape, std::tuple<int64_t, int64_t> fans, torch::TensorOptions options) {
    int64_t fan_in = std::get<0>(fans);
    int64_t fan_out = std::get<1>(fans);

    if (fan_in == -1 || fan_out == -1) {
        auto tup = compute_fans(shape);
        fan_in = std::get<0>(tup);
        fan_out = std::get<1>(tup);
    }

    float std = sqrt(2.0 / (fan_in + fan_out));

    return torch::randn(shape, options).mul_(std);
}

torch::Tensor uniform_init(float scale_factor, std::vector<int64_t> shape, torch::TensorOptions options) {
    return (2 * torch::rand(shape, options) - 1).mul_(scale_factor);
}
torch::Tensor normal_init(float mean, float std, std::vector<int64_t> shape, torch::TensorOptions options) {
    return torch::randn(shape, options).mul_(std) + mean;
}

torch::Tensor constant_init(float constant, std::vector<int64_t> shape, torch::TensorOptions options) { return torch::ones(shape, options) * constant; }

torch::Tensor initialize_tensor(shared_ptr<InitConfig> init_config, std::vector<int64_t> shape, torch::TensorOptions tensor_options,
                                std::tuple<int64_t, int64_t> fans) {
    InitDistribution init_distribution = init_config->type;
    shared_ptr<InitOptions> init_options = init_config->options;

    torch::Tensor ret;

    if (init_distribution == InitDistribution::GLOROT_NORMAL) {
        ret = glorot_normal(shape, fans, tensor_options);
    } else if (init_distribution == InitDistribution::GLOROT_UNIFORM) {
        ret = glorot_uniform(shape, fans, tensor_options);
    } else if (init_distribution == InitDistribution::UNIFORM) {
        float scale_factor = std::dynamic_pointer_cast<UniformInitOptions>(init_options)->scale_factor;
        ret = uniform_init(scale_factor, shape, tensor_options);
    } else if (init_distribution == InitDistribution::NORMAL) {
        float mean = std::dynamic_pointer_cast<NormalInitOptions>(init_options)->mean;
        float std = std::dynamic_pointer_cast<NormalInitOptions>(init_options)->std;
        ret = normal_init(mean, std, shape, tensor_options);
    } else if (init_distribution == InitDistribution::ZEROS) {
        ret = torch::zeros(shape, tensor_options);
    } else if (init_distribution == InitDistribution::ONES) {
        ret = torch::ones(shape, tensor_options);
    } else if (init_distribution == InitDistribution::CONSTANT) {
        float constant = std::dynamic_pointer_cast<ConstantInitOptions>(init_options)->constant;
        ret = constant_init(constant, shape, tensor_options);
    } else {
        throw std::runtime_error("Unimplemented initialization distribution");
    }

    return ret;
}

// Allows for initialization of small pieces of a larger tensor, for initialization methods which scale based on the tensor size
torch::Tensor initialize_subtensor(shared_ptr<InitConfig> init_config, std::vector<int64_t> sub_shape, std::vector<int64_t> full_shape,
                                   torch::TensorOptions tensor_options, std::tuple<int64_t, int64_t> fans) {
    InitDistribution init_distribution = init_config->type;
    torch::Tensor ret;

    if (init_distribution == InitDistribution::GLOROT_NORMAL) {
        if (std::get<0>(fans) == -1 || std::get<1>(fans) == -1) {
            fans = compute_fans(full_shape);
        }
        ret = glorot_normal(sub_shape, fans, tensor_options);
    } else if (init_distribution == InitDistribution::GLOROT_UNIFORM) {
        if (std::get<0>(fans) == -1 || std::get<1>(fans) == -1) {
            fans = compute_fans(full_shape);
        }
        ret = glorot_uniform(sub_shape, fans, tensor_options);
    } else {
        ret = initialize_tensor(init_config, sub_shape, tensor_options);
    }

    return ret;
}