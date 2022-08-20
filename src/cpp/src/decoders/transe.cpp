//
// Created by Jason Mohoney on 9/29/21.
//

#include "decoders/transe.h"

TransE::TransE(int num_relations, int embedding_size, torch::TensorOptions tensor_options, bool use_inverse_relations) {
    comparator_ = new L2Compare();
    relation_operator_ = new TranslationOperator();
    num_relations_ = num_relations;
    embedding_size_ = embedding_size;
    use_inverse_relations_ = use_inverse_relations;
    tensor_options_ = tensor_options;

    TransE::reset();
}

void TransE::reset() {
    relations_ = torch::zeros({num_relations_, embedding_size_}, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
    if (use_inverse_relations_) {
        inverse_relations_ = torch::zeros({num_relations_, embedding_size_}, tensor_options_).set_requires_grad(true);
        inverse_relations_ = register_parameter("inverse_relation_embeddings", inverse_relations_);
    }
}
