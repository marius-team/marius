//
// Created by Jason Mohoney on 2019-11-20.
//

#ifndef MARIUS_EMBEDDING_H
#define MARIUS_EMBEDDING_H

#include "batch.h"
#include "datatypes.h"

using std::tuple;

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


// Embedding Comparator Functions

class Comparator {
  public:
    virtual ~Comparator() {};
    virtual tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) = 0;
};

class CosineCompare : public Comparator {
  public:
    CosineCompare() {};

    tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override;
};

class DotCompare : public Comparator {
  public:
    DotCompare() {};

    tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override;
};

// Loss Functions Functions
class LossFunction {
  public:
    virtual ~LossFunction() {};
    virtual torch::Tensor operator()(const torch::Tensor &pos_scores, const torch::Tensor &neg_scores) = 0;
};

class SoftMax : public LossFunction {
  public:
    SoftMax() {};

    torch::Tensor operator()(const torch::Tensor &pos_scores, const torch::Tensor &neg_scores) override;
};

class RankingLoss : public LossFunction {
  private:
    float margin_;
  public:
    RankingLoss(float margin) {
        margin_ = margin;
    };

    torch::Tensor operator()(const torch::Tensor &pos_scores, const torch::Tensor &neg_scores) override;
};

// Decoder Models
class Decoder {
  public:
    virtual ~Decoder() { };
    virtual void forward(Batch *, bool train) = 0;
};


class LinkPredictionDecoder : public Decoder {
  protected:
    Comparator *comparator_;
    RelationOperator *relation_operator_;
    LossFunction *loss_function_;
  public:
    LinkPredictionDecoder();

    LinkPredictionDecoder(Comparator *comparator, RelationOperator *relation_operator, LossFunction *loss_function);

    void forward(Batch *, bool train) override;
};

class DistMult : public LinkPredictionDecoder {
  public:
    DistMult();
};

class TransE : public LinkPredictionDecoder {
  public:
    TransE();
};

class ComplEx : public LinkPredictionDecoder {
  public:
    ComplEx();
};

class NodeClassificationDecoder : public Decoder {
  public:
    NodeClassificationDecoder();

    void forward(Batch *, bool train) override {};
};

class RelationClassificationDecoder : public Decoder {};

#endif //MARIUS_EMBEDDING_H
