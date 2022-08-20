//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_LOSS_H_
#define MARIUS_SRC_CPP_INCLUDE_LOSS_H_

#include "configuration/config.h"
#include "datatypes.h"

// Loss Functions

/**
  Calculates loss for generated embeddings. Currently only supports link prediction losses. Node classification is hard-coded to use torch.cross_entropy.
*/
class LossFunction {
  public:
    virtual ~LossFunction() {};
    /**
      Takes positive and negative scores and calculates loss.
      @param pos_scores Positive scores
      @param neg_scores Negative scores
      @return Loss vector
    */
    virtual torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) = 0;
};

class SoftMax : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    SoftMax(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
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

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEAfterSigmoidLoss: public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    BCEAfterSigmoidLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEWithLogitsLoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    BCEWithLogitsLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class MSELoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    MSELoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class SoftPlusLoss : public LossFunction {
  private:
    LossReduction reduction_type_;
  public:
    SoftPlusLoss(shared_ptr<LossOptions> options) {
        reduction_type_ = options->loss_reduction;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

shared_ptr<LossFunction> getLossFunction(shared_ptr<LossConfig> config);

#endif //MARIUS_SRC_CPP_INCLUDE_LOSS_H_
