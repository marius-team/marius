//
// Created by Jason Mohoney on 2/5/22.
//

#ifndef MARIUS_NODE_DECODER_H
#define MARIUS_NODE_DECODER_H

#include "nn/decoders/decoder.h"

class NodeDecoder : public Decoder {
   public:
    virtual torch::Tensor forward(torch::Tensor node_repr) = 0;
};

#endif  // MARIUS_NODE_DECODER_H
