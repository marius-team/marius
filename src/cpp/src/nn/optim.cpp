//
// Created by Jason Mohoney on 12/9/21.
//

#include "nn/optim.h"

void Optimizer::load(torch::serialize::InputArchive &input_archive) {
    torch::IValue tmp;
    input_archive.read("num_steps", tmp);
    num_steps_ = tmp.toInt();

    for (auto itr = state_dict_.begin(); itr != state_dict_.end(); itr++) {
        std::string key = itr->key();
        torch::OrderedDict<std::string, torch::Tensor> param_state = torch::OrderedDict<std::string, torch::Tensor>();

        torch::serialize::InputArchive tmp_archive;
        input_archive.read(key, tmp_archive);

        for (auto itr2 = itr->value().begin(); itr2 != itr->value().end(); itr2++) {
            tmp_archive.read(itr2->key(), state_dict_[key][itr2->key()]);
        }
    }
}

void Optimizer::save(torch::serialize::OutputArchive &output_archive) {
    output_archive.write("num_steps", num_steps_);

    for (auto itr = state_dict_.begin(); itr != state_dict_.end(); itr++) {
        std::string key = itr->key();
        torch::OrderedDict<std::string, torch::Tensor> param_state = torch::OrderedDict<std::string, torch::Tensor>();

        torch::serialize::OutputArchive tmp_archive;

        for (auto itr2 = itr->value().begin(); itr2 != itr->value().end(); itr2++) {
            tmp_archive.write(itr2->key(), itr2->value());
        }

        output_archive.write(key, tmp_archive);
    }
}

void Optimizer::clear_grad() {
    auto param_items = param_dict_.items();
#pragma omp parallel for
    for (int i = 0; i < param_dict_.size(); i++) {
        param_items[i].value().mutable_grad() = torch::Tensor();
    }
}

SGDOptimizer::SGDOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, float learning_rate) {
    param_dict_ = param_dict;
    learning_rate_ = learning_rate;

    reset_state();
}

void SGDOptimizer::reset_state() { num_steps_ = 0; }

void SGDOptimizer::step() {
    num_steps_++;

    auto param_items = param_dict_.items();
#pragma omp parallel for
    for (int i = 0; i < param_dict_.size(); i++) {
        torch::NoGradGuard no_grad;

        std::string key = param_items[i].key();
        torch::Tensor param = param_items[i].value();
        torch::Tensor param_grad = param.grad();

        if (!param_grad.defined()) {
            continue;
        }

        double learning_rate = learning_rate_;

        param.data().add_(-learning_rate * param_grad);
    }
}

std::shared_ptr<Optimizer> SGDOptimizer::clone() { return std::make_shared<SGDOptimizer>(*this); }

AdagradOptimizer::AdagradOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, std::shared_ptr<AdagradOptions> options) {
    param_dict_ = param_dict;

    learning_rate_ = options->learning_rate;
    eps_ = options->eps;
    lr_decay_ = options->lr_decay;
    weight_decay_ = options->weight_decay;
    init_value_ = options->init_value;

    reset_state();
}

void AdagradOptimizer::reset_state() {
    num_steps_ = 0;
    state_dict_ = torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>>();

    for (auto itr = param_dict_.begin(); itr != param_dict_.end(); itr++) {
        std::string key = itr->key();

        torch::OrderedDict<std::string, torch::Tensor> param_state = torch::OrderedDict<std::string, torch::Tensor>();

        torch::Tensor sum_state = torch::zeros_like(itr->value());

        if (init_value_ != 0) {
            sum_state.fill_(init_value_);
        }
        param_state.insert("sum", sum_state);
        state_dict_.insert(key, param_state);
    }
}

void AdagradOptimizer::step() {
    auto param_items = param_dict_.items();
#pragma omp parallel for
    for (int i = 0; i < param_dict_.size(); i++) {
        torch::NoGradGuard no_grad;

        std::string key = param_items[i].key();
        torch::Tensor param = param_items[i].value();
        torch::Tensor param_grad = param.grad();

        if (!param_grad.defined()) {
            continue;
        }

        torch::Tensor sum_state = state_dict_[key]["sum"];

        if (weight_decay_ != 0) {
            param_grad = param_grad.add(param, weight_decay_);
        }

        double learning_rate = learning_rate_;
        if (lr_decay_ != 0) {
            learning_rate = learning_rate / (1 + num_steps_ * lr_decay_);
        }

        sum_state.addcmul_(param_grad, param_grad, 1.0);
        const auto std = sum_state.sqrt().add_(eps_);
        param.data().addcdiv_(param_grad, std, -learning_rate);
    }

    num_steps_++;
}

std::shared_ptr<Optimizer> AdagradOptimizer::clone() { return std::make_shared<AdagradOptimizer>(*this); }

AdamOptimizer::AdamOptimizer(torch::OrderedDict<std::string, torch::Tensor> param_dict, std::shared_ptr<AdamOptions> options) {
    param_dict_ = param_dict;

    learning_rate_ = options->learning_rate;
    eps_ = options->eps;
    beta_1_ = options->beta_1;
    beta_2_ = options->beta_2;
    weight_decay_ = options->weight_decay;
    amsgrad_ = options->amsgrad;

    reset_state();
}

void AdamOptimizer::reset_state() {
    num_steps_ = 0;
    state_dict_ = torch::OrderedDict<std::string, torch::OrderedDict<std::string, torch::Tensor>>();

    for (auto itr = param_dict_.begin(); itr != param_dict_.end(); itr++) {
        std::string key = itr->key();

        torch::OrderedDict<std::string, torch::Tensor> param_state = torch::OrderedDict<std::string, torch::Tensor>();

        torch::Tensor exp_avg_state = torch::zeros_like(itr->value());
        torch::Tensor exp_avg_sq_state = torch::zeros_like(itr->value());

        param_state.insert("exp_avg", exp_avg_state);
        param_state.insert("exp_avg_sq", exp_avg_sq_state);

        if (amsgrad_) {
            torch::Tensor max_exp_avg_sq_state = torch::zeros_like(itr->value());
            param_state.insert("max_exp_avg_sq", max_exp_avg_sq_state);
        }

        state_dict_.insert(key, param_state);
    }
}

void AdamOptimizer::step() {
    auto param_items = param_dict_.items();
#pragma omp parallel for
    for (int i = 0; i < param_dict_.size(); i++) {
        torch::NoGradGuard no_grad;

        std::string key = param_items[i].key();
        torch::Tensor param = param_items[i].value();
        torch::Tensor param_grad = param.grad();

        if (!param_grad.defined()) {
            continue;
        }

        torch::Tensor exp_avg_state = state_dict_[key]["exp_avg"];
        torch::Tensor exp_avg_sq_state = state_dict_[key]["exp_avg_sq"];

        float bias_correction1 = 1 - std::pow(beta_1_, num_steps_ + 1);
        float bias_correction2 = 1 - std::pow(beta_2_, num_steps_ + 1);

        if (weight_decay_ != 0) {
            param_grad = param_grad.add(param, weight_decay_);
        }

        // Decay the first and second moment running average coefficient
        exp_avg_state.mul_(beta_1_).add_(param_grad, 1 - beta_1_);
        exp_avg_sq_state.mul_(beta_2_).addcmul_(param_grad, param_grad, 1 - beta_2_);

        torch::Tensor denom;
        if (amsgrad_) {
            torch::Tensor max_exp_avg_sq_state = state_dict_[key]["max_exp_avg_sq"];
            // Maintains the maximum of all 2nd moment running avg. till now
            torch::max_out(max_exp_avg_sq_state, exp_avg_sq_state, max_exp_avg_sq_state);

            // Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq_state.sqrt() / sqrt(bias_correction2)).add_(eps_);
        } else {
            denom = (exp_avg_sq_state.sqrt() / sqrt(bias_correction2)).add_(eps_);
        }

        auto step_size = learning_rate_ / bias_correction1;

        param.data().addcdiv_(exp_avg_state, denom, -step_size);
    }

    num_steps_++;
}

std::shared_ptr<Optimizer> AdamOptimizer::clone() { return std::make_shared<AdamOptimizer>(*this); }
