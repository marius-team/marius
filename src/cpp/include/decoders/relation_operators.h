//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_RELATION_OPERATOR_H
#define MARIUS_RELATION_OPERATOR_H

#include "datatypes.h"

// Relation Operators
class RelationOperator {
public:
    virtual ~RelationOperator() {};
    virtual Embeddings operator()(const Embeddings &embs, const Relations &rels) = 0;
};

class HadamardOperator : public RelationOperator {
public:
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override;
};

class ComplexHadamardOperator : public RelationOperator {
public:
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override;
};

class TranslationOperator : public RelationOperator {
public:
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override;
};

class NoOp : public RelationOperator {
public:
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override;
};

#endif //MARIUS_RELATION_OPERATOR_H
