//
// Created by Jason Mohoney on 8/24/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_REPORTING_H_
#define MARIUS_SRC_CPP_INCLUDE_REPORTING_H_

#include "common/datatypes.h"

class Metric {
   public:
    std::string name_;
    std::string unit_;

    virtual ~Metric(){};
};

class RankingMetric : public Metric {
   public:
    virtual torch::Tensor computeMetric(torch::Tensor ranks) = 0;
};

class HitskMetric : public RankingMetric {
    int k_;

   public:
    HitskMetric(int k);

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class MeanRankMetric : public RankingMetric {
   public:
    MeanRankMetric();

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class MeanReciprocalRankMetric : public RankingMetric {
   public:
    MeanReciprocalRankMetric();

    torch::Tensor computeMetric(torch::Tensor ranks);
};

class ClassificationMetric : public Metric {
   public:
    virtual torch::Tensor computeMetric(torch::Tensor y_true, torch::Tensor y_pred) = 0;
};

class CategoricalAccuracyMetric : public ClassificationMetric {
   public:
    CategoricalAccuracyMetric();

    torch::Tensor computeMetric(torch::Tensor y_true, torch::Tensor y_pred) override;
};

class Reporter {
   private:
    std::mutex *lock_;

   public:
    std::vector<shared_ptr<Metric>> metrics_;

    Reporter() { lock_ = new std::mutex(); }

    virtual ~Reporter();

    void lock() { lock_->lock(); }

    void unlock() { lock_->unlock(); }

    void addMetric(shared_ptr<Metric> metric) { metrics_.emplace_back(metric); }

    virtual void report() = 0;
};

class LinkPredictionReporter : public Reporter {
   public:
    std::vector<torch::Tensor> per_batch_ranks_;
    std::vector<torch::Tensor> per_batch_scores_;
    std::vector<torch::Tensor> per_batch_edges_;
    torch::Tensor all_ranks_;
    torch::Tensor all_scores_;
    torch::Tensor all_edges_;

    LinkPredictionReporter();

    ~LinkPredictionReporter();

    void clear();

    torch::Tensor computeRanks(torch::Tensor pos_scores, torch::Tensor neg_scores);

    void addResult(torch::Tensor pos_scores, torch::Tensor neg_scores, torch::Tensor edges = torch::Tensor());

    void report() override;

    void save(string directory, bool scores, bool ranks);
};

class NodeClassificationReporter : public Reporter {
   public:
    std::vector<torch::Tensor> per_batch_y_true_;
    std::vector<torch::Tensor> per_batch_y_pred_;
    std::vector<torch::Tensor> per_batch_nodes_;
    torch::Tensor all_y_true_;
    torch::Tensor all_y_pred_;
    torch::Tensor all_nodes_;

    NodeClassificationReporter();

    ~NodeClassificationReporter();

    void clear();

    void addResult(torch::Tensor y_true, torch::Tensor y_pred, torch::Tensor node_ids = torch::Tensor());

    void report() override;

    void save(string directory, bool labels);
};

class ProgressReporter : public Reporter {
    std::string item_name_;
    int64_t total_items_;
    int64_t current_item_;
    int total_reports_;
    int64_t next_report_;
    int64_t items_per_report_;

   public:
    ProgressReporter(std::string item_name, int64_t total_items, int total_reports);

    ~ProgressReporter();

    void clear();

    void addResult(int64_t items_processed);

    void report() override;
};

#endif  // MARIUS_SRC_CPP_INCLUDE_REPORTING_H_
