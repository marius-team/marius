//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_TRANSE_H
#define MARIUS_TRANSE_H

#include "decoders/decoder.h"

class TransE : public LinkPredictionDecoder, public torch::nn::Cloneable<TransE> {
public:
    TransE(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations=true);

    void reset() override;
};

#endif //MARIUS_TRANSE_H
