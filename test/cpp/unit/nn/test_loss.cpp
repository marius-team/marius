//
// Created by Jason Mohoney on 2/4/22.
//

#include <gtest/gtest.h>
#include <nn/loss.h>

torch::Tensor test_pos1 = torch::tensor({500.0}, torch::kFloat32);
torch::Tensor test_pos2 = torch::tensor({.1}, torch::kFloat32);
torch::Tensor test_pos3 = torch::tensor({-500.0}, torch::kFloat32);
torch::Tensor test_pos4 = torch::tensor({.5, 2.5, 5.0, 7.5, 100.0, 250.0}, torch::kFloat32);

torch::Tensor test_neg1 = torch::tensor({{150.0, 100.0, 50.0, 25.0, 10.0}}, torch::kFloat32);
torch::Tensor test_neg2 = torch::tensor({{.001, -.001, -.005, -.1, -10.0}}, torch::kFloat32);
torch::Tensor test_neg3 = torch::tensor({{-150.0, -100.0, -50.0, -25.0, 10.0}}, torch::kFloat32);
torch::Tensor test_neg4 = torch::tensor({{.5, 10.0}, {2.5, -1.0}, {5.0, 1.0}, {7.5, -5.0}, {100.0, 20.0}, {250.0, 10.0}}, torch::kFloat32);

torch::Tensor test_y_pred1 = torch::tensor({{.75, .25}, {.5, .5}, {3.0, .25}}, torch::kFloat32);
torch::Tensor test_y_label1 = torch::tensor({0, 1, 0}, torch::kInt64);
// torch::Tensor test_y_label1 = torch::tensor({{.75, .25}, {.5, .5}, {.9, .1}}, torch::kFloat32);

torch::Tensor test_y_pred2 = torch::tensor({{.75, .25, .1}, {.5, .5, .9}, {3.0, .25, 5.0}}, torch::kFloat32);
torch::Tensor test_y_label2 = torch::tensor({0, 2, 2}, torch::kInt64);
// torch::Tensor test_y_label2 = torch::tensor({{.75, .20, .05}, {.2, .2, .6}, {.35, .05, .6}}, torch::kFloat32);

torch::Tensor invalid_tensor = torch::tensor({{{0.0}}}, torch::kFloat32);

torch::Tensor undef_tensor;

TEST(TestLoss, TestShapeMismatch) {
    // check undefined pos
    EXPECT_THROW(check_score_shapes(undef_tensor, test_neg4), UndefinedTensorException);

    // check undefined neg
    EXPECT_THROW(check_score_shapes(test_pos1, undef_tensor), UndefinedTensorException);

    // check invalid pos
    EXPECT_THROW(check_score_shapes(test_neg4, test_neg4), TensorSizeMismatchException);
    EXPECT_THROW(check_score_shapes(invalid_tensor, test_neg4), TensorSizeMismatchException);

    // check invalid neg
    EXPECT_THROW(check_score_shapes(test_pos4, test_pos4), TensorSizeMismatchException);
    EXPECT_THROW(check_score_shapes(test_pos4, invalid_tensor), TensorSizeMismatchException);

    // check neg mismatch
    EXPECT_THROW(check_score_shapes(test_pos1, test_neg4), TensorSizeMismatchException);

    // check valid
    EXPECT_NO_THROW(check_score_shapes(test_pos1, test_neg1));
    EXPECT_NO_THROW(check_score_shapes(test_pos2, test_neg2));
    EXPECT_NO_THROW(check_score_shapes(test_pos3, test_neg3));
    EXPECT_NO_THROW(check_score_shapes(test_pos4, test_neg4));
}

TEST(TestLoss, TestSoftmaxCrossEntropy) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new SoftmaxCrossEntropy(loss_options_mean);
    auto *loss_fn_sum = new SoftmaxCrossEntropy(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false), MariusRuntimeException);
    ASSERT_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false), MariusRuntimeException);

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false), MariusRuntimeException);
    ASSERT_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false), MariusRuntimeException);

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / test_pos4.size(0)) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestRankingLoss) {
    auto loss_options_mean = std::make_shared<RankingLossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;
    loss_options_mean->margin = 0.0;

    auto loss_options_sum = std::make_shared<RankingLossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;
    loss_options_sum->margin = 0.0;

    auto *loss_fn_mean = new RankingLoss(loss_options_mean);
    auto *loss_fn_sum = new RankingLoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false), MariusRuntimeException);

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false), MariusRuntimeException);

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4);

    ASSERT_TRUE(((sum_loss / (test_pos4.size(0) * 2)) == mean_loss).all().item<bool>());

    // test margin
    float margin1 = -10.0;
    float margin2 = 5.0;
    float margin3 = 10.0;

    loss_options_sum->margin = margin1;
    auto *loss_fn_sum1 = new RankingLoss(loss_options_sum);

    auto loss1 = loss_fn_sum1->operator()(test_pos4, test_neg4);

    loss_options_sum->margin = margin2;
    auto *loss_fn_sum2 = new RankingLoss(loss_options_sum);

    auto loss2 = loss_fn_sum2->operator()(test_pos4, test_neg4);

    loss_options_sum->margin = margin3;
    auto *loss_fn_sum3 = new RankingLoss(loss_options_sum);

    auto loss3 = loss_fn_sum3->operator()(test_pos4, test_neg4);

    ASSERT_TRUE((loss1 < loss2).all().item<bool>());
    ASSERT_TRUE((loss2 < loss3).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
    delete loss_fn_sum1;
    delete loss_fn_sum2;
    delete loss_fn_sum3;
}

TEST(TestLoss, TestCrossEntropyLoss) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new CrossEntropyLoss(loss_options_mean);
    auto *loss_fn_sum = new CrossEntropyLoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false));

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false));

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / (test_pos4.size(0))) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestBCEAfterSigmoid) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new BCEAfterSigmoidLoss(loss_options_mean);
    auto *loss_fn_sum = new BCEAfterSigmoidLoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false));

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false));

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / (3 * test_pos4.size(0))) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestBCEWithLogits) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new BCEWithLogitsLoss(loss_options_mean);
    auto *loss_fn_sum = new BCEWithLogitsLoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false));

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false));

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / (3 * test_pos4.size(0))) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestMSE) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new MSELoss(loss_options_mean);
    auto *loss_fn_sum = new MSELoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false));

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false));

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / (3 * test_pos4.size(0))) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestSoftPlus) {
    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto loss_options_sum = std::make_shared<LossOptions>();
    loss_options_sum->loss_reduction = LossReduction::SUM;

    auto *loss_fn_mean = new SoftPlusLoss(loss_options_mean);
    auto *loss_fn_sum = new SoftPlusLoss(loss_options_sum);

    // test mean reduction
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_mean->operator()(test_y_pred2, test_y_label2, false));

    // test sum reduction
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos1, test_neg1, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos2, test_neg2, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos3, test_neg3, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_pos4, test_neg4, true));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred1, test_y_label1, false));
    ASSERT_NO_THROW(loss_fn_sum->operator()(test_y_pred2, test_y_label2, false));

    auto mean_loss = loss_fn_mean->operator()(test_pos4, test_neg4, true);
    auto sum_loss = loss_fn_sum->operator()(test_pos4, test_neg4, true);

    ASSERT_TRUE(((sum_loss / (3 * test_pos4.size(0))) == mean_loss).all().item<bool>());

    delete loss_fn_mean;
    delete loss_fn_sum;
}

TEST(TestLoss, TestGetLossFunction) {
    // test nullptr
    shared_ptr<LossConfig> loss_config = nullptr;
    EXPECT_THROW(getLossFunction(loss_config), UnexpectedNullPtrException);

    auto loss_options_mean = std::make_shared<LossOptions>();
    loss_options_mean->loss_reduction = LossReduction::MEAN;

    auto ranking_loss_options_mean = std::make_shared<RankingLossOptions>();
    ranking_loss_options_mean->loss_reduction = LossReduction::MEAN;
    ranking_loss_options_mean->margin = 1.0;

    // test softmax
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::SOFTMAX_CE;
    loss_config->options = loss_options_mean;

    auto softmax_loss = new SoftmaxCrossEntropy(loss_options_mean);
    auto ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(softmax_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete softmax_loss;

    // test ranking
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::RANKING;
    loss_config->options = ranking_loss_options_mean;

    auto ranking_loss = new RankingLoss(ranking_loss_options_mean);
    ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(ranking_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete ranking_loss;

    // test bce sigmoid
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::BCE_AFTER_SIGMOID;
    loss_config->options = loss_options_mean;

    auto bce_sigmoid_loss = new BCEAfterSigmoidLoss(loss_options_mean);
    ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(bce_sigmoid_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete bce_sigmoid_loss;

    // test bce logits
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::BCE_WITH_LOGITS;
    loss_config->options = loss_options_mean;

    auto bce_logits_loss = new BCEWithLogitsLoss(loss_options_mean);
    ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(bce_logits_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete bce_logits_loss;

    // test mse
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::MSE;
    loss_config->options = loss_options_mean;

    auto mse_loss = new MSELoss(loss_options_mean);
    ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(mse_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete mse_loss;

    // test softplus
    loss_config = std::make_shared<LossConfig>();
    loss_config->type = LossFunctionType::SOFTPLUS;
    loss_config->options = loss_options_mean;

    auto softplus_loss = new SoftPlusLoss(loss_options_mean);
    ret_loss = getLossFunction(loss_config);
    ASSERT_EQ(softplus_loss->operator()(test_pos4, test_neg4, true).item<float>(), ret_loss->operator()(test_pos4, test_neg4, true).item<float>());

    delete softplus_loss;
}