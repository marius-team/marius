//
// Created by Jason Mohoney on 12/9/21.
//

#ifndef MARIUS_OPTIM_H
#define MARIUS_OPTIM_H

#include "common/datatypes.h"
#include "configuration/config.h"

class Optimizer {
   public:
    int64_t num_steps_;

    torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>> state_dict_;
    torch::OrderedDict<std::string, torch::Tensor> param_dict_;

    virtual ~Optimizer(){};

    void save(torch::serialize::OutputArchive &output_archive);

    void load(torch::serialize::InputArchive &input_archive);

    void clear_grad();

    virtual void reset_state() = 0;

    virtual void step() = 0;

    virtual std::shared_ptr<Optimizer> clone() = 0;
};

class SGDOptimizer : public Optimizer {
   public:
    float learning_rate_;

    SGDOptimizer(const SGDOptimizer &optim) {
        param_dict_ = optim.param_dict_;
        learning_rate_ = optim.learning_rate_;
        reset_state();
    }

    SGDOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, float learning_rate);

    void reset_state() override;

    void step() override;

    std::shared_ptr<Optimizer> clone() override;
};

class AdagradOptimizer : public Optimizer {
   public:
    float learning_rate_;
    float eps_;
    float lr_decay_;
    float weight_decay_;
    float init_value_;

    AdagradOptimizer(const AdagradOptimizer &optim) {
        param_dict_ = optim.param_dict_;
        learning_rate_ = optim.learning_rate_;
        eps_ = optim.eps_;
        lr_decay_ = optim.lr_decay_;
        weight_decay_ = optim.weight_decay_;
        init_value_ = optim.init_value_;
        reset_state();
    }

    AdagradOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, std::shared_ptr<AdagradOptions> options);

    void reset_state() override;

    void step() override;

    std::shared_ptr<Optimizer> clone() override;
};

class AdamOptimizer : public Optimizer {
   public:
    float learning_rate_;
    float eps_;
    float beta_1_;
    float beta_2_;
    float weight_decay_;
    bool amsgrad_;

    AdamOptimizer(const AdamOptimizer &optim) {
        param_dict_ = optim.param_dict_;
        learning_rate_ = optim.learning_rate_;
        eps_ = optim.eps_;
        beta_1_ = optim.beta_1_;
        beta_2_ = optim.beta_2_;
        weight_decay_ = optim.weight_decay_;
        amsgrad_ = optim.amsgrad_;
        reset_state();
    }

    AdamOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, std::shared_ptr<AdamOptions> options);

    void reset_state() override;

    void step() override;

    std::shared_ptr<Optimizer> clone() override;
};

#endif  // MARIUS_OPTIM_H
