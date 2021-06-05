#include <loss.h>
#include <iostream>

torch::Tensor SoftMax::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    auto scores = torch::cat({pos_scores.unsqueeze(1), neg_scores.logsumexp(1, true)}, 1);
    torch::nn::functional::CrossEntropyFuncOptions options;
    options.reduction(torch::kSum);
    auto loss = torch::nn::functional::cross_entropy(scores, pos_scores.new_zeros({}, device_options).expand(pos_scores.size(0)), options);

    return loss;
}

torch::Tensor RankingLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    torch::nn::functional::MarginRankingLossFuncOptions options;
    options.reduction(torch::kSum);
    options.margin(margin_);
    auto loss = torch::nn::functional::margin_ranking_loss(neg_scores, pos_scores.unsqueeze(1), pos_scores.new_full({1, 1}, -1, device_options), options);
    return loss;
}

torch::Tensor BCEAfterSigmoidLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    neg_scores = neg_scores.flatten(0, 1);
    auto scores = torch::cat({pos_scores, neg_scores});
    auto labels = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)});
    torch::nn::functional::BinaryCrossEntropyFuncOptions options;
    options.reduction(torch::kSum);
    auto loss = torch::nn::functional::binary_cross_entropy(scores.sigmoid(), labels, options);
    return loss;
}

torch::Tensor BCEWithLogitsLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device()); 
    neg_scores = neg_scores.flatten(0, 1);
    auto scores = torch::cat({pos_scores, neg_scores});
    auto labels = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)});
    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions options;
    options.reduction(torch::kSum);
    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(scores, labels, options);
    return loss; 
}

torch::Tensor MSELoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    neg_scores = neg_scores.flatten(0, 1);
    auto scores = torch::cat({pos_scores, neg_scores});
    auto labels = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)});
    torch::nn::functional::MSELossFuncOptions options;
    options.reduction(torch::kSum);
    auto loss = torch::nn::functional::mse_loss(scores, labels, options);
    return loss;
}

torch::Tensor SoftPlusLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    neg_scores = neg_scores.flatten(0, 1);
    auto scores = torch::cat({pos_scores, neg_scores});
    auto target = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)});
    target = 2 * target - 1;
    auto loss = torch::nn::functional::softplus(((-1) * target * scores));
    loss = loss.mean();
    return loss;
}