//
// Created by Jason Mohoney on 9/17/21.
//

#ifndef MARIUS_MODEL_HELPERS_H
#define MARIUS_MODEL_HELPERS_H

#include "model.h"

template <typename T>
void decoder_clone_helper(const std::shared_ptr<Model>& primary_model, const std::shared_ptr<Model>& cloned_model, torch::Device device) {
    std::shared_ptr<T> decoder = std::dynamic_pointer_cast<T>(primary_model->decoder_);
    cloned_model->decoder_ = std::dynamic_pointer_cast<T>(decoder->clone(device));
    cloned_model->register_module("decoder", std::dynamic_pointer_cast<T>(cloned_model->decoder_));
    cloned_model->decoder_optimizer_ = getOptimizerForModule(std::dynamic_pointer_cast<T>(cloned_model->decoder_), primary_model->model_config_->decoder->optimizer);
}

template <typename T>
void encoder_clone_helper(const std::shared_ptr<Model>& primary_model, const std::shared_ptr<Model>& cloned_model, torch::Device device) {
    std::shared_ptr<T> encoder = std::dynamic_pointer_cast<T>(primary_model->encoder_);
    cloned_model->encoder_ = std::dynamic_pointer_cast<T>(encoder->clone(device));
    cloned_model->register_module("encoder", std::dynamic_pointer_cast<T>(cloned_model->encoder_));
    cloned_model->encoder_optimizer_ = getOptimizerForModule(std::dynamic_pointer_cast<T>(cloned_model->encoder_), primary_model->model_config_->encoder->optimizer);
}


#endif //MARIUS_MODEL_HELPERS_H
