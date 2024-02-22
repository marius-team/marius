//
// Created by Jason Mohoney on 3/31/22.
//

#ifndef MARIUS_DECODER_METHODS_H
#define MARIUS_DECODER_METHODS_H

#include "common/datatypes.h"
#include "nn/decoders/edge/edge_decoder.h"

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges, torch::Tensor node_embeddings);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs = torch::Tensor());

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids);

#endif  // MARIUS_DECODER_METHODS_H
