//
// Created by Jason Mohoney on 8/25/21.
//
#include "nn/regularizer.h"

NormRegularizer::NormRegularizer(int norm, float coefficient) {
    norm_ = norm;
    coefficient_ = coefficient;
}

torch::Tensor NormRegularizer::operator()(torch::Tensor src_nodes_embs, torch::Tensor dst_node_embs) {
    return coefficient_ / 2 * torch::sum((torch::norm(src_nodes_embs, norm_, 0) + torch::norm(dst_node_embs, norm_, 0)));
}