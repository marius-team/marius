//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_LOSS_H_
#define MARIUS_SRC_CPP_INCLUDE_LOSS_H_

#include "common/datatypes.h"
#include "configuration/config.h"

void check_score_shapes(torch::Tensor pos_scores, torch::Tensor neg_scores);

std::tuple<torch::Tensor, torch::Tensor> scores_to_labels(torch::Tensor pos_scores, torch::Tensor neg_scores, bool one_hot);

torch::Tensor to_one_hot(torch::Tensor labels, int num_classes);

// Loss Functions
/**
  Calculates loss for generated embeddings. Currently only supports link prediction losses. Node classification is hard-coded to use torch.cross_entropy.
*/
class LossFunction {
   public:
    virtual ~LossFunction(){};
    /**
      Takes positive and negative scores and calculates loss.
      @param pos_scores Positive scores
      @param neg_scores Negative scores
      @return Loss vector
    */
    virtual torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) = 0;
};

class SoftmaxCrossEntropy : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    SoftmaxCrossEntropy(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

class RankingLoss : public LossFunction {
   private:
    float margin_;
    LossReduction reduction_type_;

   public:
    RankingLoss(shared_ptr<RankingLossOptions> options) {
        margin_ = options->margin;
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores, bool scores = true) override;
};

class CrossEntropyLoss : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    CrossEntropyLoss(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

class BCEAfterSigmoidLoss : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    BCEAfterSigmoidLoss(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

class BCEWithLogitsLoss : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    BCEWithLogitsLoss(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

class MSELoss : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    MSELoss(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

class SoftPlusLoss : public LossFunction {
   private:
    LossReduction reduction_type_;

   public:
    SoftPlusLoss(shared_ptr<LossOptions> options) { reduction_type_ = options->loss_reduction; };

    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override;
};

shared_ptr<LossFunction> getLossFunction(shared_ptr<LossConfig> config);

#endif  // MARIUS_SRC_CPP_INCLUDE_LOSS_H_
