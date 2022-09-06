//
// Created by Jason Mohoney on 2/1/22.
//

#ifndef MARIUS_FEATURE_H
#define MARIUS_FEATURE_H

#include "common/datatypes.h"
#include "nn/layers/layer.h"

class FeatureLayer : public Layer {
   public:
    int offset_;

    FeatureLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int offset = 0);

    torch::Tensor forward(torch::Tensor input);

    void reset() override;
};

#endif  // MARIUS_FEATURE_H
