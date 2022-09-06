//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_COMPLEX_H
#define MARIUS_COMPLEX_H

#include "nn/decoders/edge/edge_decoder.h"

class ComplEx : public EdgeDecoder, public torch::nn::Cloneable<ComplEx> {
   public:
    ComplEx(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(), bool use_inverse_relations = true,
            EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;
};

#endif  // MARIUS_COMPLEX_H
