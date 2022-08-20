//
// Created by Jason Mohoney on 9/29/21.
//

#include <decoders/relation_operators.h>

Embeddings HadamardOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs * rels;
}

Embeddings ComplexHadamardOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    int dim = embs.size(1);

    int real_len = dim / 2;
    int imag_len = dim - dim / 2;

    Embeddings real_emb = embs.narrow(1, 0, real_len);
    Embeddings imag_emb = embs.narrow(1, real_len, imag_len);

    Relations real_rel = rels.narrow(1, 0, real_len);
    Relations imag_rel = rels.narrow(1, real_len, imag_len);

    Embeddings out = torch::zeros_like(embs);

    out.narrow(1, 0, real_len) = (real_emb * real_rel) - (imag_emb * imag_rel);
    out.narrow(1, real_len, imag_len) = (real_emb * imag_rel) + (imag_emb * real_rel);

    return out;
}

Embeddings TranslationOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs + rels;
}

Embeddings NoOp::operator()(const Embeddings &embs, const Relations &rels) {
    (void) rels;
    return embs;
}
