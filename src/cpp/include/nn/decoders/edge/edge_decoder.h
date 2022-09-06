//
// Created by Jason Mohoney on 2/6/22.
//

#ifndef MARIUS_EDGE_DECODER_H
#define MARIUS_EDGE_DECODER_H

#include "common/datatypes.h"
#include "nn/decoders/decoder.h"
#include "nn/decoders/edge/comparators.h"
#include "nn/decoders/edge/relation_operators.h"

class EdgeDecoder : public Decoder {
   public:
    shared_ptr<Comparator> comparator_;
    shared_ptr<RelationOperator> relation_operator_;
    torch::Tensor relations_;
    torch::Tensor inverse_relations_;
    int num_relations_;
    int embedding_size_;
    torch::TensorOptions tensor_options_;
    EdgeDecoderMethod decoder_method_;

    bool use_inverse_relations_;

    torch::Tensor apply_relation(torch::Tensor nodes, torch::Tensor relations);

    torch::Tensor compute_scores(torch::Tensor src, torch::Tensor dst);

    torch::Tensor select_relations(torch::Tensor indices, bool inverse = false);
};
#endif  // MARIUS_EDGE_DECODER_H
