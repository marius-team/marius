//
// Created by Jason Mohoney on 12/10/21.
//

#ifndef MARIUS_CONCAT_H
#define MARIUS_CONCAT_H

#include "common/datatypes.h"
#include "reduction_layer.h"

class ConcatReduction : public ReductionLayer {
   public:
    ConcatReduction(shared_ptr<LayerConfig> layer_config, torch::Device device);

    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;

    void reset() override;
};

#endif  // MARIUS_CONCAT_H
