//
// Created by Jason Mohoney on 2/7/22.
//
#include "nn/decoders/node/noop_node_decoder.h"

torch::Tensor NoOpNodeDecoder::forward(torch::Tensor nodes) { return nodes; }

void NoOpNodeDecoder::reset() { return; }
