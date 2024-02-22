//
// Created by Jason Mohoney on 8/24/21.
//
#include "reporting/reporting.h"

#include <fstream>

#include "configuration/constants.h"
#include "reporting/logger.h"

HitskMetric::HitskMetric(int k) {
    k_ = k;
    name_ = "Hits@" + std::to_string(k_);
    unit_ = "";
}

torch::Tensor HitskMetric::computeMetric(torch::Tensor ranks) { return torch::tensor((double)ranks.le(k_).nonzero().size(0) / ranks.size(0), torch::kFloat64); }

MeanRankMetric::MeanRankMetric() {
    name_ = "Mean Rank";
    unit_ = "";
}

torch::Tensor MeanRankMetric::computeMetric(torch::Tensor ranks) { return ranks.to(torch::kFloat64).mean(); }

MeanReciprocalRankMetric::MeanReciprocalRankMetric() {
    name_ = "MRR";
    unit_ = "";
}

torch::Tensor MeanReciprocalRankMetric::computeMetric(torch::Tensor ranks) { return ranks.to(torch::kFloat32).reciprocal().mean(); }

CategoricalAccuracyMetric::CategoricalAccuracyMetric() {
    name_ = "Accuracy";
    unit_ = "%";
}

torch::Tensor CategoricalAccuracyMetric::computeMetric(torch::Tensor y_true, torch::Tensor y_pred) {
    return 100 * torch::tensor({(double)(y_true == y_pred).nonzero().size(0) / y_true.size(0)}, torch::kFloat64);
}

Reporter::~Reporter() { delete lock_; }

LinkPredictionReporter::LinkPredictionReporter() {}

LinkPredictionReporter::~LinkPredictionReporter() { clear(); }

void LinkPredictionReporter::clear() {
    all_ranks_ = torch::Tensor();
    per_batch_ranks_ = {};
    per_batch_scores_ = {};
    all_scores_ = torch::Tensor();
}

torch::Tensor LinkPredictionReporter::computeRanks(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    return (neg_scores >= pos_scores.unsqueeze(1)).sum(1) + 1;
}

void LinkPredictionReporter::addResult(torch::Tensor pos_scores, torch::Tensor neg_scores, torch::Tensor edges) {
    lock();

    if (neg_scores.defined()) {
        per_batch_ranks_.emplace_back(computeRanks(pos_scores, neg_scores));
    }

    if (edges.defined()) {
        per_batch_scores_.emplace_back(pos_scores.to(torch::kCPU));
        per_batch_edges_.emplace_back(edges.to(torch::kCPU));
    }
    unlock();
}

void LinkPredictionReporter::report() {
    all_ranks_ = torch::cat(per_batch_ranks_).to(torch::kCPU);
    if (per_batch_scores_.size() > 0) {
        all_scores_ = torch::cat(per_batch_scores_);
    }
    per_batch_ranks_ = {};
    per_batch_scores_ = {};

    std::string report_string = "";
    std::string header = "\n=================================\nLink Prediction: " + std::to_string(all_ranks_.size(0)) + " edges evaluated\n";
    report_string = report_string + header;

    std::string tmp;
    for (auto m : metrics_) {
        torch::Tensor result = std::dynamic_pointer_cast<RankingMetric>(m)->computeMetric(all_ranks_);
        tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
        report_string = report_string + tmp;
    }
    std::string footer = "=================================";
    report_string = report_string + footer;

    SPDLOG_INFO(report_string);
}

void LinkPredictionReporter::save(string directory, bool scores, bool ranks) {
    all_ranks_ = torch::cat(per_batch_ranks_).to(torch::kCPU);
    if (per_batch_scores_.size() > 0) {
        all_scores_ = torch::cat(per_batch_scores_);
    }
    per_batch_ranks_ = {};
    per_batch_scores_ = {};

    if (!metrics_.empty()) {
        std::string report_string = "";
        std::string header = "Link Prediction: " + std::to_string(all_ranks_.size(0)) + " edges evaluated\n";
        report_string = report_string + header;

        std::string tmp;
        for (auto m : metrics_) {
            torch::Tensor result = std::dynamic_pointer_cast<RankingMetric>(m)->computeMetric(all_ranks_);
            tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
            report_string = report_string + tmp;
        }

        string metrics_file = directory + PathConstants::output_metrics_file;

        std::ofstream metrics_stream;
        metrics_stream.open(metrics_file);

        metrics_stream << report_string;
        metrics_stream.close();
    }

    if (ranks || scores) {
        if (per_batch_edges_.empty()) {
            throw MariusRuntimeException("To save scores or ranks, the evaluated edges must be provided to addResult()");
        }

        all_edges_ = torch::cat({per_batch_edges_});

        string output_scores_file = directory + PathConstants::output_scores_file;
        std::ofstream scores_stream;
        scores_stream.open(output_scores_file);

        string header_string = "";
        std::vector<torch::Dtype> dtypes;
        if (all_edges_.size(1) == 3) {
            header_string = "src,rel,dst";
            dtypes = {torch::kInt64, torch::kInt64, torch::kInt64};
        } else {
            header_string = "src,dst";
            dtypes = {torch::kInt64, torch::kInt64};
        }

        torch::Tensor output_tensor = all_edges_.to(torch::kFloat32);
        if (ranks) {
            output_tensor = torch::cat({output_tensor, all_ranks_.narrow(0, 0, all_edges_.size(0)).to(torch::kFloat32).unsqueeze(1)}, 1);
            header_string = header_string + ",rank";
            dtypes.emplace_back(torch::kInt64);
        }

        if (scores) {
            output_tensor = torch::cat({output_tensor, all_scores_.narrow(0, 0, all_edges_.size(0)).to(torch::kFloat32).unsqueeze(1)}, 1);
            header_string = header_string + ",score";
            dtypes.emplace_back(torch::kFloat32);
        }

        scores_stream << header_string << "\n";
        auto accessor = output_tensor.accessor<float, 2>();

        int64_t num_rows = output_tensor.size(0);
        int64_t num_cols = output_tensor.size(1);
        for (int64_t row = 0; row < num_rows; row++) {
            string row_string = "";
            for (int64_t col = 0; col < num_cols - 1; col++) {
                row_string = row_string + std::to_string((int)accessor[row][col]) + ",";
            }

            if (scores) {
                row_string = row_string + std::to_string(accessor[row][num_cols - 1]) + "\n";
            } else {
                row_string = row_string + std::to_string((int)accessor[row][num_cols - 1]) + "\n";
            }

            scores_stream << row_string;
        }
        scores_stream.close();
    }
}

NodeClassificationReporter::NodeClassificationReporter() {}

NodeClassificationReporter::~NodeClassificationReporter() { clear(); }

void NodeClassificationReporter::clear() {
    all_y_true_ = torch::Tensor();
    all_y_pred_ = torch::Tensor();
    per_batch_y_true_ = {};
    per_batch_y_pred_ = {};
}

void NodeClassificationReporter::addResult(torch::Tensor y_true, torch::Tensor y_pred, torch::Tensor node_ids) {
    lock();
    per_batch_y_true_.emplace_back(y_true);
    per_batch_y_pred_.emplace_back(y_pred.argmax(1));

    if (node_ids.defined()) {
        per_batch_nodes_.emplace_back(node_ids);
    }
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
        torch::Tensor result = std::dynamic_pointer_cast<ClassificationMetric>(m)->computeMetric(all_y_true_, all_y_pred_);
        tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
        report_string = report_string + tmp;
    }
    std::string footer = "=================================";
    report_string = report_string + footer;

    SPDLOG_INFO(report_string);
}

void NodeClassificationReporter::save(string directory, bool labels) {
    all_y_true_ = torch::cat(per_batch_y_true_).to(torch::kCPU);
    all_y_pred_ = torch::cat(per_batch_y_pred_).to(torch::kCPU);
    per_batch_y_true_ = {};
    per_batch_y_pred_ = {};

    if (!metrics_.empty()) {
        std::string report_string = "";
        std::string header = "\n=================================\nNode Classification: " + std::to_string(all_y_true_.size(0)) + " nodes evaluated\n";
        report_string = report_string + header;

        std::string tmp;
        for (auto m : metrics_) {
            torch::Tensor result = std::dynamic_pointer_cast<ClassificationMetric>(m)->computeMetric(all_y_true_, all_y_pred_);
            tmp = m->name_ + ": " + std::to_string(result.item<double>()) + m->unit_ + "\n";
            report_string = report_string + tmp;
        }
        std::string footer = "=================================";
        report_string = report_string + footer;

        string metrics_file = directory + PathConstants::output_metrics_file;

        std::ofstream metrics_stream;
        metrics_stream.open(metrics_file);

        metrics_stream << report_string;
        metrics_stream.close();
    }

    if (labels) {
        if (per_batch_nodes_.empty()) {
            throw MariusRuntimeException("To save labels, the evaluated node ids must be provided to add_result()");
        }

        all_nodes_ = torch::cat({per_batch_nodes_}).to(torch::kCPU);

        string output_labels_file = directory + PathConstants::output_labels_file;
        std::ofstream labels_stream;
        labels_stream.open(output_labels_file);

        string header_string = "id,y_pred,y_true";

        torch::Tensor output_tensor = all_nodes_.to(torch::kFloat32).unsqueeze(1);
        output_tensor = torch::cat({output_tensor, all_y_pred_.to(torch::kFloat32).unsqueeze(1)}, 1);
        output_tensor = torch::cat({output_tensor, all_y_true_.to(torch::kFloat32).unsqueeze(1)}, 1);

        labels_stream << header_string << "\n";
        auto accessor = output_tensor.accessor<float, 2>();

        int64_t num_rows = output_tensor.size(0);
        int64_t num_cols = output_tensor.size(1);
        for (int64_t row = 0; row < num_rows; row++) {
            string row_string = "";
            for (int64_t col = 0; col < num_cols - 1; col++) {
                row_string = row_string + std::to_string((int)accessor[row][col]) + ",";
            }
            row_string = row_string + std::to_string((int)accessor[row][num_cols - 1]) + "\n";

            labels_stream << row_string;
        }
        labels_stream.close();
    }
}

ProgressReporter::ProgressReporter(std::string item_name, int64_t total_items, int total_reports) {
    item_name_ = item_name;
    total_items_ = total_items;
    current_item_ = 0;
    total_reports_ = total_reports;
    items_per_report_ = total_items_ / total_reports_;
    next_report_ = items_per_report_;
}

ProgressReporter::~ProgressReporter() { clear(); }

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
    std::string report_string = item_name_ + " processed: [" + std::to_string(current_item_) + "/" + std::to_string(total_items_) + "], " +
                                fmt::format("{:.2f}", 100 * (double)current_item_ / total_items_) + "%";
    SPDLOG_INFO(report_string);
}
