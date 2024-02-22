//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/decoders/edge/distmult.h"

DistMult::DistMult(int num_relations, int embedding_size, torch::TensorOptions tensor_options, bool use_inverse_relations, EdgeDecoderMethod decoder_method) {
    comparator_ = std::make_shared<DotCompare>();
    relation_operator_ = std::make_shared<HadamardOperator>();
    num_relations_ = num_relations;
    embedding_size_ = embedding_size;
    use_inverse_relations_ = use_inverse_relations;
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;

    learning_task_ = LearningTask::LINK_PREDICTION;

    DistMult::reset();
}

void DistMult::reset() {
    relations_ = torch::ones({num_relations_, embedding_size_}, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
    if (use_inverse_relations_) {
        inverse_relations_ = torch::ones({num_relations_, embedding_size_}, tensor_options_).set_requires_grad(true);
        inverse_relations_ = register_parameter("inverse_relation_embeddings", inverse_relations_);
    }
}