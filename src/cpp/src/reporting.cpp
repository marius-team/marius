//
// Created by Jason Mohoney on 8/24/21.
//

#include "logger.h"
#include "reporting.h"

HitskMetric::HitskMetric(int k) {
    k_ = k;
    name_ = "Hits@" + std::to_string(k_);
    unit_ = "";
}

torch::Tensor HitskMetric::computeMetric(torch::Tensor ranks) {
    return torch::tensor((double) ranks.le(k_).nonzero().size(0) / ranks.size(0), torch::kFloat64);
}

MeanRankMetric::MeanRankMetric() {
    name_ = "Mean Rank";
    unit_ = "";
}

torch::Tensor MeanRankMetric::computeMetric(torch::Tensor ranks) {
    return ranks.to(torch::kFloat64).mean();
}

MeanReciprocalRankMetric::MeanReciprocalRankMetric() {
    name_ = "MRR";
    unit_ = "";
}

torch::Tensor MeanReciprocalRankMetric::computeMetric(torch::Tensor ranks) {
    return ranks.to(torch::kFloat32).reciprocal().mean();
}

CategoricalAccuracyMetric::CategoricalAccuracyMetric() {
    name_ = "Accuracy";
    unit_ = "%";
}

torch::Tensor CategoricalAccuracyMetric::computeMetric(torch::Tensor y_true, torch::Tensor y_pred) {
    return 100 * torch::tensor({(double) (y_true == y_pred).nonzero().size(0) / y_true.size(0)}, torch::kFloat64);
}

Reporter::~Reporter() {
    delete lock_;
}

LinkPredictionReporter::LinkPredictionReporter() {

}

LinkPredictionReporter::~LinkPredictionReporter() {
    clear();
}

void LinkPredictionReporter::clear() {
    all_ranks_ = torch::Tensor();
    per_batch_ranks_ = {};
    per_batch_results_ = {};
}

torch::Tensor LinkPredictionReporter::computeRanks(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    return (neg_scores >= pos_scores.unsqueeze(1)).sum(1) + 1;
}

void LinkPredictionReporter::addResult(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    lock();
    per_batch_ranks_.emplace_back(computeRanks(pos_scores, neg_scores));
    unlock();
}

void LinkPredictionReporter::report() {
    all_ranks_ = torch::cat(per_batch_ranks_);
    per_batch_ranks_ = {};

    std::string report_string = "";
    std::string header = "\n=================================\nLink Prediction: " + std::to_string(all_ranks_.size(0)) + " edges evaluated\n";
    report_string = report_string + header;

    std::string tmp;
    for (auto m : metrics_) {
        torch::Tensor result = ((RankingMetric *) m)->computeMetric(all_ranks_);
        tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
        report_string = report_string + tmp;
    }
    std::string footer = "=================================";
    report_string = report_string + footer;

    SPDLOG_INFO(report_string);
}

NodeClassificationReporter::NodeClassificationReporter() {

}

NodeClassificationReporter::~NodeClassificationReporter() {
    clear();
}

void NodeClassificationReporter::clear() {
    all_y_true_ = torch::Tensor();
    all_y_pred_ = torch::Tensor();
    per_batch_y_true_ = {};
    per_batch_y_pred_ = {};
}

void NodeClassificationReporter::addResult(torch::Tensor y_true, torch::Tensor y_pred) {
    lock();
    per_batch_y_true_.emplace_back(y_true);
    per_batch_y_pred_.emplace_back(y_pred.argmax(1));
    unlock();
}

void NodeClassificationReporter::report() {
    all_y_true_ = torch::cat(per_batch_y_true_);
    all_y_pred_ = torch::cat(per_batch_y_pred_);
    per_batch_y_true_ = {};
    per_batch_y_pred_ = {};

    std::string report_string = "";
    std::string header = "\n=================================\nNode Classification: " + std::to_string(all_y_true_.size(0)) + " nodes evaluated\n";
    report_string = report_string + header;

    std::string tmp;
    for (auto m : metrics_) {
        torch::Tensor result = ((ClassificationMetric *) m)->computeMetric(all_y_true_, all_y_pred_);
        tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
        report_string = report_string + tmp;
    }
    std::string footer = "=================================";
    report_string = report_string + footer;

    SPDLOG_INFO(report_string);
}

ProgressReporter::ProgressReporter(std::string item_name, int64_t total_items, int total_reports) {
    item_name_ = item_name;
    total_items_ = total_items;
    current_item_ = 0;
    total_reports_ = total_reports;
    items_per_report_ = total_items_ / total_reports_;
    next_report_ = items_per_report_;
}

ProgressReporter::~ProgressReporter() {
    clear();
}

void ProgressReporter::clear() {
    current_item_ = 0;
    next_report_ = items_per_report_;
}

void ProgressReporter::addResult(int64_t items_processed) {
    lock();
    current_item_ += items_processed;
    if (current_item_ >= next_report_) {
        report();
        next_report_ = std::min({current_item_ + items_per_report_, total_items_});
    }
    unlock();
}

void ProgressReporter::report() {
    std::string report_string = item_name_ + " processed: [" + std::to_string(current_item_) + "/" + std::to_string(total_items_) + "], " + std::to_string(100 * (double) current_item_ / total_items_) + "%";
    SPDLOG_INFO(report_string);
}