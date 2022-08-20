//
// Created by Jason Mohoney on 9/29/21.
//

#include "decoders/decoder.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Decoder::forward(Batch *, bool train) {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ret;
    return ret;
}

EmptyDecoder::EmptyDecoder() {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LinkPredictionDecoder::forward(Batch *batch, bool train) {

    torch::Tensor lhs_pos_scores;
    torch::Tensor lhs_neg_scores;
    torch::Tensor rhs_pos_scores;
    torch::Tensor rhs_neg_scores;

    // localSample
    batch->localSample();

    // corrupt destination
    Embeddings adjusted_src_pos = batch->src_pos_embeddings_;
    if (batch->rel_indices_.defined()) {
        adjusted_src_pos = (*relation_operator_)(batch->src_pos_embeddings_, relations_.index_select(0, batch->rel_indices_));
    }
    std::tie(rhs_pos_scores, rhs_neg_scores) = (*comparator_)(adjusted_src_pos, batch->dst_pos_embeddings_, batch->dst_all_neg_embeddings_);

    // corrupt source (if using double sided relations
    if (use_inverse_relations_) {
        Embeddings adjusted_dst_pos = batch->dst_pos_embeddings_;
        if (batch->rel_indices_.defined()) {
            adjusted_dst_pos = (*relation_operator_)(batch->dst_pos_embeddings_, inverse_relations_.index_select(0, batch->rel_indices_));
        }
        std::tie(lhs_pos_scores, lhs_neg_scores) = (*comparator_)(adjusted_dst_pos, batch->src_pos_embeddings_, batch->src_all_neg_embeddings_);
    }

    // filter scores
    if (batch->dst_neg_filter_.defined()) {
        rhs_neg_scores.flatten(0, 1).index_fill_(0, batch->dst_neg_filter_, -1e9);
        if (lhs_neg_scores.defined()) {
            lhs_neg_scores.flatten(0, 1).index_fill_(0, batch->src_neg_filter_, -1e9);
        }
    }
    return std::forward_as_tuple(rhs_pos_scores, rhs_neg_scores, lhs_pos_scores, lhs_neg_scores);
}
