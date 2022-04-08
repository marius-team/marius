//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/decoders/edge/complex.h"

ComplEx::ComplEx(int num_relations, int embedding_size, torch::TensorOptions tensor_options, bool use_inverse_relations, EdgeDecoderMethod decoder_method) {
    comparator_ = std::make_shared<CosineCompare>();
    relation_operator_ = std::make_shared<TranslationOperator>();
    num_relations_ = num_relations;
    embedding_size_ = embedding_size;
    use_inverse_relations_ = use_inverse_relations;
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;

    learning_task_ = LearningTask::LINK_PREDICTION;

    ComplEx::reset();
}

void ComplEx::reset() {
    relations_ = torch::zeros({num_relations_, embedding_size_}, tensor_options_);
    relations_.narrow(1, 0, (embedding_size_ / 2) - 1).fill_(1);
    relations_ = register_parameter("relation_embeddings", relations_).set_requires_grad(true);
    if (use_inverse_relations_) {
        inverse_relations_ = torch::zeros({num_relations_, embedding_size_}, tensor_options_);
        inverse_relations_.narrow(1, 0, (embedding_size_ / 2) - 1).fill_(1).set_requires_grad(true);
        inverse_relations_ = register_parameter("inverse_relation_embeddings", inverse_relations_);
    }
}