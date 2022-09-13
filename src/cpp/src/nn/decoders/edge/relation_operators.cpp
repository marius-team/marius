//
// Created by Jason Mohoney on 9/29/21.
//

#include <nn/decoders/edge/relation_operators.h>

torch::Tensor HadamardOperator::operator()(const torch::Tensor &embs, const torch::Tensor &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs * rels;
}

torch::Tensor ComplexHadamardOperator::operator()(const torch::Tensor &embs, const torch::Tensor &rels) {
    if (!rels.defined()) {
        return embs;
    }
    int dim = embs.size(1);

    int real_len = dim / 2;
    int imag_len = dim - dim / 2;

    torch::Tensor real_emb = embs.narrow(1, 0, real_len);
    torch::Tensor imag_emb = embs.narrow(1, real_len, imag_len);

    torch::Tensor real_rel = rels.narrow(1, 0, real_len);
    torch::Tensor imag_rel = rels.narrow(1, real_len, imag_len);

    torch::Tensor out = torch::zeros_like(embs);

    out.narrow(1, 0, real_len) = (real_emb * real_rel) - (imag_emb * imag_rel);
    out.narrow(1, real_len, imag_len) = (real_emb * imag_rel) + (imag_emb * real_rel);

    return out;
}

torch::Tensor TranslationOperator::operator()(const torch::Tensor &embs, const torch::Tensor &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs + rels;
}

torch::Tensor NoOp::operator()(const torch::Tensor &embs, const torch::Tensor &rels) {
    (void)rels;
    return embs;
}
