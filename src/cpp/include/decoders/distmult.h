//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_DISTMULT_H
#define MARIUS_DISTMULT_H

#include "decoders/decoder.h"

class DistMult : public LinkPredictionDecoder, public torch::nn::Cloneable<DistMult> {
public:
    DistMult(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations=true);

    void reset() override;
};

#endif //MARIUS_DISTMULT_H
