//
// Created by Jason Mohoney on 2/4/22.
//

#include <gtest/gtest.h>
#include <nn/activation.h>

torch::Tensor tensor1 = torch::tensor({-1.0, -5.2, 1.2, 3.5, 5.5, 7.0}, torch::kFloat32);
torch::Tensor tensor2 = torch::tensor({1.0, 5.2, 1.2, 3.5, 5.5, 7.0}, torch::kFloat32);
torch::Tensor tensor3 = torch::tensor({1.0}, torch::kFloat32);
torch::Tensor tensor4 = torch::tensor({-1.0}, torch::kFloat32);
torch::Tensor tensor5 = torch::tensor({{-1.0, -5.2, 1.2, 3.5, 5.5, 7.0}, {1.0, 5.2, 1.2, 3.5, 5.5, 7.0}}, torch::kFloat32);
torch::Tensor empty_tensor;

TEST(TestActivation, TestRelu) {
    ActivationFunction activation = ActivationFunction::RELU;

    torch::Tensor expected_tensor1 = torch::tensor({0.0, 0.0, 1.2, 3.5, 5.5, 7.0}, torch::kFloat32);
    torch::Tensor expected_tensor2 = torch::tensor({1.0, 5.2, 1.2, 3.5, 5.5, 7.0}, torch::kFloat32);
    torch::Tensor expected_tensor3 = torch::tensor({1.0}, torch::kFloat32);
    torch::Tensor expected_tensor4 = torch::tensor({0.0}, torch::kFloat32);
    torch::Tensor expected_tensor5 = torch::tensor({{0.0, 0.0, 1.2, 3.5, 5.5, 7.0}, {1.0, 5.2, 1.2, 3.5, 5.5, 7.0}}, torch::kFloat32);

    ASSERT_TRUE(apply_activation(activation, tensor1).eq(expected_tensor1).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor2).eq(expected_tensor2).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor3).eq(expected_tensor3).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor4).eq(expected_tensor4).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor5).eq(expected_tensor5).all().item<bool>());

    ASSERT_THROW(apply_activation(activation, empty_tensor), UndefinedTensorException);
}

TEST(TestActivation, TestSigmoid) {
    ActivationFunction activation = ActivationFunction::SIGMOID;

    torch::Tensor expected_tensor1 = torch::sigmoid(tensor1);
    torch::Tensor expected_tensor2 = torch::sigmoid(tensor2);
    torch::Tensor expected_tensor3 = torch::sigmoid(tensor3);
    torch::Tensor expected_tensor4 = torch::sigmoid(tensor4);
    torch::Tensor expected_tensor5 = torch::sigmoid(tensor5);

    ASSERT_TRUE(apply_activation(activation, tensor1).eq(expected_tensor1).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor2).eq(expected_tensor2).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor3).eq(expected_tensor3).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor4).eq(expected_tensor4).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor5).eq(expected_tensor5).all().item<bool>());

    ASSERT_THROW(apply_activation(activation, empty_tensor), UndefinedTensorException);
}

TEST(TestActivation, TestNone) {
    ActivationFunction activation = ActivationFunction::NONE;

    ASSERT_TRUE(apply_activation(activation, tensor1).eq(tensor1).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor2).eq(tensor2).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor3).eq(tensor3).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor4).eq(tensor4).all().item<bool>());
    ASSERT_TRUE(apply_activation(activation, tensor5).eq(tensor5).all().item<bool>());

    ASSERT_THROW(apply_activation(activation, empty_tensor), UndefinedTensorException);
}
