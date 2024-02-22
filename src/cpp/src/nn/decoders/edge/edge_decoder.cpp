//
// Created by Jason Mohoney on 2/6/22.
//

#include "nn/decoders/edge/edge_decoder.h"

torch::Tensor EdgeDecoder::apply_relation(torch::Tensor nodes, torch::Tensor relations) { return relation_operator_->operator()(nodes, relations); }

torch::Tensor EdgeDecoder::compute_scores(torch::Tensor src, torch::Tensor dst) { return comparator_->operator()(src, dst); }

torch::Tensor EdgeDecoder::select_relations(torch::Tensor indices, bool inverse) {
    if (inverse) {
        if (!inverse_relations_.defined()) {
            throw UndefinedTensorException();
        }
        return inverse_relations_.index_select(0, indices);
    } else {
        return relations_.index_select(0, indices);
    }
}
