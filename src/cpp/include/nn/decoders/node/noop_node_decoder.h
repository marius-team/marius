//
// Created by Jason Mohoney on 2/7/22.
//

#ifndef MARIUS_NOOP_NODE_DECODER_H
#define MARIUS_NOOP_NODE_DECODER_H

#include "nn/decoders/node/node_decoder.h"

class NoOpNodeDecoder : public NodeDecoder, public torch::nn::Cloneable<NoOpNodeDecoder> {
   public:
    NoOpNodeDecoder() { learning_task_ = LearningTask::NODE_CLASSIFICATION; };

    torch::Tensor forward(torch::Tensor node_repr) override;

    void reset() override;
};

#endif  // MARIUS_NOOP_NODE_DECODER_H