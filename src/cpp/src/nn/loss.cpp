//
// Created by Jason Mohoney on 8/25/21.
//

#include "nn/loss.h"

void check_score_shapes(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    if (!pos_scores.defined()) {
        throw UndefinedTensorException();
    }

    if (!neg_scores.defined()) {
        throw UndefinedTensorException();
    }

    if (pos_scores.sizes().size() != 1) {
        throw TensorSizeMismatchException(pos_scores, "Positive scores should be 1-dimensional");
    }

    if (neg_scores.sizes().size() != 2) {
        throw TensorSizeMismatchException(neg_scores, "Negative scores should be 2-dimensional");
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        //        throw TensorSizeMismatchException(pos_scores, (std::stringstream("Size: ") << neg_scores.size(1) << " First dimension of pos_scores and
        //        neg_scores should match.").str());
        throw TensorSizeMismatchException(pos_scores, "First dimension of pos_scores and neg_scores should match.");
    }
}

torch::Tensor to_one_hot(torch::Tensor labels, int num_classes) {
    torch::Tensor one_hot_encodings = torch::zeros({labels.size(0), num_classes}, torch::kInt64);
    one_hot_encodings.index_fill_(1, labels.to(torch::kInt64), 1);
    return one_hot_encodings.to(torch::kFloat32);
}

std::tuple<torch::Tensor, torch::Tensor> scores_to_labels(torch::Tensor pos_scores, torch::Tensor neg_scores, bool one_hot) {
    torch::Tensor y_pred = torch::cat({pos_scores, neg_scores}, -1);
    torch::Tensor labels;
    if (one_hot) {
        labels = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)}, -1);
    } else {
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
        labels = torch::zeros({pos_scores.size(0)}, options);
    }

    return std::forward_as_tuple(y_pred, labels);
}

torch::Tensor SoftmaxCrossEntropy::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (!scores) {
        throw MariusRuntimeException(
            "Input to SoftmaxCrossEntropy loss function must be scores. SoftmaxCrossEntropy is currently unsupported for classification.");
    }

    check_score_shapes(y_pred, labels);
    std::tie(y_pred, labels) = scores_to_labels(y_pred.unsqueeze(1), labels.logsumexp(1, true), false);

    torch::nn::functional::CrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::cross_entropy(y_pred, labels, options);
}

torch::Tensor RankingLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores, bool scores) {
    // does this loss make sense?

    if (!scores) {
        throw MariusRuntimeException("Input to ranking loss function must be scores. This loss function is unsupported for classification.");
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    torch::nn::functional::MarginRankingLossFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }
    options.margin(margin_);

    return torch::nn::functional::margin_ranking_loss(neg_scores, pos_scores.unsqueeze(1), pos_scores.new_full({1, 1}, -1, device_options), options);
}

torch::Tensor CrossEntropyLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred.unsqueeze(1), labels, false);
    }

    torch::nn::functional::CrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::cross_entropy(y_pred, labels, options);
}

torch::Tensor BCEAfterSigmoidLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::BinaryCrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::binary_cross_entropy(y_pred.sigmoid(), labels, options);
}

torch::Tensor BCEWithLogitsLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::binary_cross_entropy_with_logits(y_pred, labels, options);
}

torch::Tensor MSELoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::MSELossFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::mse_loss(y_pred, labels, options);
}

torch::Tensor SoftPlusLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    labels = 2 * labels - 1;
    auto loss = torch::nn::functional::softplus(((-1) * labels * y_pred));
    if (reduction_type_ == LossReduction::MEAN) {
        loss = loss.mean();
    } else if (reduction_type_ == LossReduction::SUM) {
        loss = loss.sum();
    }

    return loss;
}

std::shared_ptr<LossFunction> getLossFunction(shared_ptr<LossConfig> config) {
    if (config == nullptr) {
        throw UnexpectedNullPtrException();
    }

    if (config->type == LossFunctionType::SOFTMAX_CE) {
        return std::make_shared<SoftmaxCrossEntropy>(config->options);
    } else if (config->type == LossFunctionType::RANKING) {
        return std::make_shared<RankingLoss>(std::dynamic_pointer_cast<RankingLossOptions>(config->options));
    } else if (config->type == LossFunctionType::CROSS_ENTROPY) {
        return std::make_shared<CrossEntropyLoss>(config->options);
    } else if (config->type == LossFunctionType::BCE_AFTER_SIGMOID) {
        return std::make_shared<BCEAfterSigmoidLoss>(config->options);
    } else if (config->type == LossFunctionType::BCE_WITH_LOGITS) {
        return std::make_shared<BCEWithLogitsLoss>(config->options);
    } else if (config->type == LossFunctionType::MSE) {
        return std::make_shared<MSELoss>(config->options);
    } else if (config->type == LossFunctionType::SOFTPLUS) {
        return std::make_shared<SoftPlusLoss>(config->options);
    } else {
        throw std::runtime_error("Unsupported loss function type");
    }
}