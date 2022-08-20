//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_COMPARATOR_H
#define MARIUS_COMPARATOR_H

#include "datatypes.h"

// Embedding Comparator Functions
class Comparator {
public:
    virtual ~Comparator() {};
    virtual std::tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) = 0;
};

class L2Compare : public Comparator {
public:
    L2Compare() {};

    std::tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override;
};

class CosineCompare : public Comparator {
public:
    CosineCompare() {};

    std::tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override;
};

class DotCompare : public Comparator {
public:
    DotCompare() {};

    std::tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override;
};

#endif //MARIUS_COMPARATOR_H
