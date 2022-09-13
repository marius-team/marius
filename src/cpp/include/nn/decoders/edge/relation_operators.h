//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_RELATION_OPERATOR_H
#define MARIUS_RELATION_OPERATOR_H

#include "common/datatypes.h"

// Relation Operators
class RelationOperator {
   public:
    virtual ~RelationOperator(){};
    virtual torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) = 0;
};

class HadamardOperator : public RelationOperator {
   public:
    torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) override;
};

class ComplexHadamardOperator : public RelationOperator {
   public:
    torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) override;
};

class TranslationOperator : public RelationOperator {
   public:
    torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) override;
};

class NoOp : public RelationOperator {
   public:
    torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) override;
};

#endif  // MARIUS_RELATION_OPERATOR_H
