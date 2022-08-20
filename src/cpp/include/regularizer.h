//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_REGULARIZER_H_
#define MARIUS_SRC_CPP_INCLUDE_REGULARIZER_H_

#include "datatypes.h"

class Regularizer {
  public:
    virtual ~Regularizer() {};

    virtual torch::Tensor operator()(Embeddings src_nodes_embs, Embedding dst_node_embs) = 0;
};

class NormRegularizer : public Regularizer {
  private:
    int norm_;
    float coefficient_;
  public:
    NormRegularizer(int norm, float coefficient);

    torch::Tensor operator()(Embeddings src_nodes_embs, Embedding dst_node_embs) override;
};

#endif //MARIUS_SRC_CPP_INCLUDE_REGULARIZER_H_
