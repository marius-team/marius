//
// Created by Jason Mohoney on 2/4/22.
//

#include <gtest/gtest.h>
#include <nn/initialization.h>

auto f16_options = torch::TensorOptions().dtype(torch::kFloat16);
auto f32_options = torch::TensorOptions().dtype(torch::kFloat32);
auto f64_options = torch::TensorOptions().dtype(torch::kFloat64);

std::vector<int64_t> shape1 = {5};
std::vector<int64_t> shape2 = {5, 3};
std::vector<int64_t> shape3 = {5, 3, 2};

TEST(TestInitialization, TestUniform) {
    float scale_factor1 = 1.0;
    float scale_factor2 = 2.0;
    float scale_factor3 = .25;

    // test scale factor
    torch::Tensor tensor = uniform_init(scale_factor1, shape2, f32_options);
    ASSERT_TRUE((tensor.ge(-scale_factor1) & tensor.le(scale_factor1)).all().item<bool>());

    tensor = uniform_init(scale_factor2, shape2, f32_options);
    ASSERT_TRUE((tensor.ge(-scale_factor2) & tensor.le(scale_factor2)).all().item<bool>());

    tensor = uniform_init(scale_factor3, shape2, f32_options);
    ASSERT_TRUE((tensor.ge(-scale_factor3) & tensor.le(scale_factor3)).all().item<bool>());

    // test shape
    tensor = uniform_init(scale_factor1, shape1, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape1);

    tensor = uniform_init(scale_factor1, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);

    tensor = uniform_init(scale_factor1, shape3, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape3);

    // test tensor options
    tensor = uniform_init(scale_factor1, shape2, f16_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat16);

    tensor = uniform_init(scale_factor1, shape2, f32_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    tensor = uniform_init(scale_factor1, shape2, f64_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat64);
}

TEST(TestInitialization, TestNormal) {
    std::vector<int64_t> shape_large = {500, 500};  // large shape used to get better estimate of mean and std for normal distribution

    float mean1 = 0.0;
    float mean2 = -.5;
    float mean3 = 2.0;

    float std1 = 1.0;
    float std2 = 2.5;
    float std3 = 5.0;

    // test mean/std
    torch::Tensor tensor = normal_init(mean1, std1, shape_large, f32_options);
    ASSERT_NEAR(tensor.mean().item<float>(), mean1, .1);
    ASSERT_NEAR(tensor.std().item<float>(), std1, .1);

    tensor = normal_init(mean2, std2, shape_large, f32_options);
    ASSERT_NEAR(tensor.mean().item<float>(), mean2, .1);
    ASSERT_NEAR(tensor.std().item<float>(), std2, .1);

    tensor = normal_init(mean3, std3, shape_large, f32_options);
    ASSERT_NEAR(tensor.mean().item<float>(), mean3, .1);
    ASSERT_NEAR(tensor.std().item<float>(), std3, .1);

    // test shape
    tensor = normal_init(mean1, std1, shape1, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape1);

    tensor = normal_init(mean1, std1, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);

    tensor = normal_init(mean1, std1, shape3, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape3);

    // test tensor options
    tensor = normal_init(mean1, std1, shape1, f16_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat16);

    tensor = normal_init(mean1, std1, shape1, f32_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    tensor = normal_init(mean1, std1, shape1, f64_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat64);
}

TEST(TestInitialization, TestConstant) {
    float val1 = 0.0;
    float val2 = -.5;
    float val3 = 2.0;

    torch::Tensor tensor = constant_init(val1, shape2, f32_options);
    ASSERT_TRUE(tensor.eq(val1).all().item<bool>());

    tensor = constant_init(val2, shape2, f32_options);
    ASSERT_TRUE(tensor.eq(val2).all().item<bool>());

    tensor = constant_init(val3, shape2, f32_options);
    ASSERT_TRUE(tensor.eq(val3).all().item<bool>());

    // test shape
    tensor = constant_init(val1, shape1, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape1);

    tensor = constant_init(val1, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);

    tensor = constant_init(val1, shape3, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape3);

    // test tensor options
    tensor = constant_init(val1, shape2, f16_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat16);

    tensor = constant_init(val1, shape2, f32_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    tensor = constant_init(val1, shape2, f64_options);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat64);
}

TEST(TestInitialization, TestComputeFans) {
    std::tuple<int64_t, int64_t> output;

    // dims = 0
    std::vector<int64_t> shape = {};
    output = compute_fans(shape);

    ASSERT_EQ(std::get<0>(output), 1);
    ASSERT_EQ(std::get<1>(output), 1);

    // dims = 1
    shape = {1};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[0]);
    ASSERT_EQ(std::get<1>(output), shape[0]);

    shape = {10};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[0]);
    ASSERT_EQ(std::get<1>(output), shape[0]);

    // dims = 2
    shape = {1, 1};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[0]);
    ASSERT_EQ(std::get<1>(output), shape[1]);

    shape = {10, 5};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[0]);
    ASSERT_EQ(std::get<1>(output), shape[1]);

    // dims > 2
    shape = {1, 1, 1};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[1]);
    ASSERT_EQ(std::get<1>(output), shape[2]);

    shape = {10, 5, 3};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[1]);
    ASSERT_EQ(std::get<1>(output), shape[2]);

    shape = {15, 3, 9};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[1]);
    ASSERT_EQ(std::get<1>(output), shape[2]);

    shape = {2, 4, 6, 8, 10};
    output = compute_fans(shape);
    ASSERT_EQ(std::get<0>(output), shape[3]);
    ASSERT_EQ(std::get<1>(output), shape[4]);
}

TEST(TestInitialization, TestGlorotUniform) {
    std::tuple<int64_t, int64_t> compute_fans = {-1, -1};
    std::tuple<int64_t, int64_t> given_fans = {1, 1};

    // dims = 0
    std::vector<int64_t> shape = {};
    torch::Tensor tensor = glorot_uniform(shape, compute_fans, f32_options);
    float limit = sqrt(6.0 / (1 + 1));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    tensor = glorot_uniform(shape, given_fans, f32_options);
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {2, 2};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (2 + 2));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims = 1
    shape = {10};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    limit = sqrt(6.0 / (10 + 10));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {10, 10};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (10 + 10));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {20, 20};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (20 + 20));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims = 2
    shape = {10, 20};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    limit = sqrt(6.0 / (10 + 20));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {10, 20};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (10 + 20));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {20, 20};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (20 + 20));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims > 2
    shape = {10, 20, 10};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    limit = sqrt(6.0 / (20 + 10));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {20, 10};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (20 + 10));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {10, 10};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (10 + 10));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    shape = {10, 20, 10, 5, 3};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    limit = sqrt(6.0 / (5 + 3));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {5, 3};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (5 + 3));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);

    given_fans = {100, 50};
    tensor = glorot_uniform(shape, given_fans, f32_options);
    limit = sqrt(6.0 / (100 + 50));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
    ASSERT_TRUE(tensor.sizes() == shape);
}

TEST(TestInitialization, TestGlorotNormal) {
    std::tuple<int64_t, int64_t> compute_fans = {-1, -1};
    std::tuple<int64_t, int64_t> given_fans = {1, 1};

    // only checking shape since it's non-trivial to check if a tensor with few elements comes from the normal distribution
    // dims = 0
    std::vector<int64_t> shape = {};
    torch::Tensor tensor = glorot_normal(shape, compute_fans, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims = 1
    shape = {10};
    tensor = glorot_normal(shape, compute_fans, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims = 2
    shape = {10, 20};
    tensor = glorot_normal(shape, compute_fans, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape);

    // dims > 2
    shape = {10, 20, 10};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape);

    shape = {10, 20, 10, 5, 3};
    tensor = glorot_uniform(shape, compute_fans, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape);
}

TEST(TestInitialization, TestTensorInit) {
    torch::Tensor tensor;
    shared_ptr<InitConfig> init_config = std::make_shared<InitConfig>();

    init_config->type = InitDistribution::GLOROT_NORMAL;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    init_config->type = InitDistribution::GLOROT_UNIFORM;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    float limit = sqrt(6.0 / (shape2[0] + shape2[1]));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());

    std::tuple<int64_t, int64_t> fans = {10, 25};
    tensor = initialize_tensor(init_config, shape2, f32_options, fans);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    limit = sqrt(6.0 / (10 + 25));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());

    init_config->type = InitDistribution::UNIFORM;
    limit = .25;
    auto options = std::make_shared<UniformInitOptions>();
    options->scale_factor = limit;
    init_config->options = options;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());

    init_config->type = InitDistribution::NORMAL;
    auto normal_options = std::make_shared<NormalInitOptions>();
    init_config->options = normal_options;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    init_config->type = InitDistribution::ZEROS;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    ASSERT_TRUE(tensor.eq(0).all().item<bool>());

    init_config->type = InitDistribution::ONES;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    ASSERT_TRUE(tensor.eq(1).all().item<bool>());

    init_config->type = InitDistribution::CONSTANT;
    auto const_options = std::make_shared<ConstantInitOptions>();
    const_options->constant = .35;
    init_config->options = const_options;
    tensor = initialize_tensor(init_config, shape2, f32_options);
    ASSERT_TRUE(tensor.sizes() == shape2);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    ASSERT_TRUE(tensor.eq(.35).all().item<bool>());
}

TEST(TestInitialization, TestSubtensorInit) {
    torch::Tensor tensor;
    shared_ptr<InitConfig> init_config = std::make_shared<InitConfig>();

    std::vector<int64_t> sub_shape = {2, 3};
    std::vector<int64_t> full_shape = {4, 3};

    init_config->type = InitDistribution::GLOROT_NORMAL;
    tensor = initialize_subtensor(init_config, sub_shape, full_shape, f32_options);
    ASSERT_TRUE(tensor.sizes() == sub_shape);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);

    init_config->type = InitDistribution::GLOROT_UNIFORM;
    tensor = initialize_subtensor(init_config, sub_shape, full_shape, f32_options);
    ASSERT_TRUE(tensor.sizes() == sub_shape);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    float limit = sqrt(6.0 / (sub_shape[0] + sub_shape[1]));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());

    std::tuple<int64_t, int64_t> fans = {10, 25};
    tensor = initialize_subtensor(init_config, sub_shape, full_shape, f32_options, fans);
    ASSERT_TRUE(tensor.sizes() == sub_shape);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    limit = sqrt(6.0 / (10 + 25));
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());

    init_config->type = InitDistribution::UNIFORM;
    limit = .25;
    auto options = std::make_shared<UniformInitOptions>();
    options->scale_factor = limit;
    init_config->options = options;
    tensor = initialize_subtensor(init_config, sub_shape, full_shape, f32_options);
    ASSERT_TRUE(tensor.sizes() == sub_shape);
    ASSERT_TRUE(tensor.dtype() == torch::kFloat32);
    ASSERT_TRUE((tensor.ge(-limit) & tensor.le(limit)).all().item<bool>());
}