//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_INITIALIZATION_H
#define MARIUS_INITIALIZATION_H

#include "common/datatypes.h"
#include "configuration/config.h"

std::tuple<int64_t, int64_t> compute_fans(std::vector<int64_t> shape);

torch::Tensor glorot_uniform(std::vector<int64_t> shape, std::tuple<int64_t, int64_t> fans, torch::TensorOptions options);

torch::Tensor glorot_normal(std::vector<int64_t> shape, std::tuple<int64_t, int64_t> fans, torch::TensorOptions options);

torch::Tensor constant_init(float constant, std::vector<int64_t> shape, torch::TensorOptions options);

torch::Tensor uniform_init(float scale_factor, std::vector<int64_t> shape, torch::TensorOptions options);

torch::Tensor normal_init(float mean, float std, std::vector<int64_t> shape, torch::TensorOptions options);

torch::Tensor initialize_tensor(shared_ptr<InitConfig> init_config, std::vector<int64_t> shape, torch::TensorOptions tensor_options,
                                std::tuple<int64_t, int64_t> fans = {-1, -1});

/** For initializing large tensors that won't fit in memory */
torch::Tensor initialize_subtensor(shared_ptr<InitConfig> init_config, std::vector<int64_t> sub_shape, std::vector<int64_t> full_shape,
                                   torch::TensorOptions tensor_options, std::tuple<int64_t, int64_t> fans = {-1, -1});

#endif  // MARIUS_INITIALIZATION_H
