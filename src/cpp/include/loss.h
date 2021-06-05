#ifndef MARIUS_LOSS_H
#define MARIUS_LOSS_H

#include <torch/torch.h>

// Loss Functions Functions
class LossFunction {
  public:
    virtual ~LossFunction() {};
    virtual torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) = 0;
};

class SoftMax : public LossFunction {
  public:
    SoftMax() {};

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class RankingLoss : public LossFunction {
  private:
    float margin_;
  public:
    RankingLoss(float margin) {
        margin_ = margin;
    };

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEAfterSigmoidLoss: public LossFunction {
  public:
    BCEAfterSigmoidLoss() {};

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class BCEWithLogitsLoss : public LossFunction {
  public:
    BCEWithLogitsLoss() {};

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class MSELoss : public LossFunction {
  public:
    MSELoss() {};

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

class SoftPlusLoss : public LossFunction {
  public:
    SoftPlusLoss() {};

    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override;
};

#endif