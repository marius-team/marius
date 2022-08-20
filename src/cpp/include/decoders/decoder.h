//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_DECODER_H
#define MARIUS_DECODER_H

#include "batch.h"
#include "datatypes.h"
#include "decoders/comparators.h"
#include "decoders/relation_operators.h"


// Decoder Models
class Decoder {
public:
    virtual ~Decoder() { };

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(Batch *, bool train);
};

class EmptyDecoder : public Decoder, public torch::nn::Cloneable<EmptyDecoder> {
public:
    EmptyDecoder();

    void reset() override {};
};

class LinkPredictionDecoder : public Decoder {
protected:
    Comparator *comparator_;
    RelationOperator *relation_operator_;
    Relations relations_;
    Relations inverse_relations_;
    int num_relations_;
    int embedding_size_;
    torch::TensorOptions tensor_options_;

public:
    bool use_inverse_relations_;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(Batch *, bool train) override;
};

#endif //MARIUS_DECODER_H
