//
// Created by Jason Mohoney on 2/1/22.
//

#ifndef MARIUS_LAYER_H
#define MARIUS_LAYER_H

#include "common/datatypes.h"
#include "configuration/config.h"
#include "data/graph.h"
#include "nn/activation.h"
#include "nn/initialization.h"

class Layer : public torch::nn::Module {
   public:
    shared_ptr<LayerConfig> config_;
    torch::Device device_;
    torch::Tensor bias_;

    Layer();

    virtual ~Layer(){};

    virtual void reset() = 0;

    torch::Tensor post_hook(torch::Tensor input);

    void init_bias();
};

#endif  // MARIUS_LAYER_H
