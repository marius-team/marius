//
// Created by Jason Mohoney on 8/25/21.
//
#include "regularizer.h"

NormRegularizer::NormRegularizer(int norm, float coefficient) {
    norm_ = norm;
    coefficient_ = coefficient;
}

torch::Tensor NormRegularizer::operator()(Embeddings src_nodes_embs, Embedding dst_node_embs) {
    return coefficient_ / 2 * torch::sum((torch::norm(src_nodes_embs, norm_, 0) + torch::norm(dst_node_embs, norm_, 0)));
}