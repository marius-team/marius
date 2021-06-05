#ifndef MARIUS_LOSS_H
#define MARIUS_LOSS_H

#include <torch/torch.h>
#include <datatypes.h>

// Loss Functions Functions
class LossFunction {
  public:
    virtual ~LossFunction() {};
    virtual torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) = 0;
};

class SoftMax : public LossFunction {
  private:
    ReductionType reduction_type_;
  public:
    SoftMax(ReductionType reduction_type) {
      reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class RankingLoss : public LossFunction {
  private:
    float margin_;
    ReductionType reduction_type_;
  public:
    RankingLoss(float margin, ReductionType reduction_type) {
        margin_ = margin;
        ReductionType reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEAfterSigmoidLoss: public LossFunction {
  private:
    ReductionType reduction_type_;
  public:
    BCEAfterSigmoidLoss(ReductionType reduction_type) {
      reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEWithLogitsLoss : public LossFunction {
  private:
    ReductionType reduction_type_;
  public:
    BCEWithLogitsLoss(ReductionType reduction_type) {
      reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class MSELoss : public LossFunction {
  private:
    ReductionType reduction_type_;
  public:
    MSELoss(ReductionType reduction_type) {
      reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class SoftPlusLoss : public LossFunction {
  private:
    ReductionType reduction_type_;
  public:
    SoftPlusLoss(ReductionType reduction_type) {
      reduction_type_ = reduction_type;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

#endif