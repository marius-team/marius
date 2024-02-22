//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_DISTMULT_H
#define MARIUS_DISTMULT_H

#include "nn/decoders/edge/edge_decoder.h"

class DistMult : public EdgeDecoder, public torch::nn::Cloneable<DistMult> {
   public:
    DistMult(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations = true,
             EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;
};

#endif  // MARIUS_DISTMULT_H
