//
// Created by Jason Mohoney on 10/7/21.
//

#include "activation.h"

torch::Tensor apply_activation(ActivationFunction activation_function, torch::Tensor input) {
    if (activation_function == ActivationFunction::RELU) {
        return torch::relu(input);
    } else if (activation_function == ActivationFunction::SIGMOID) {
        return torch::sigmoid(input);
    } else if (activation_function == ActivationFunction::NONE) {
        return input;
    } else {
        throw std::runtime_error("Unimplemented activation function");
    }
}
