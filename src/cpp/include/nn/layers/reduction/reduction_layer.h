//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_FEATURIZER_H_
#define MARIUS_FEATURIZER_H_

#include "common/datatypes.h"
#include "configuration/config.h"
#include "nn/layers/layer.h"

/**
  Generates new embeddings for nodes by combining node features and their respective embeddings in order to emphasize individual node properties.
*/
class ReductionLayer : public Layer {
   public:
    virtual ~ReductionLayer(){};

    virtual torch::Tensor forward(std::vector<torch::Tensor> inputs) = 0;
};

#endif  // MARIUS_FEATURIZER_H_
