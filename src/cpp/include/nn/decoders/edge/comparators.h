//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_COMPARATOR_H
#define MARIUS_COMPARATOR_H

#include "common/datatypes.h"

torch::Tensor pad_and_reshape(torch::Tensor input, int num_chunks);

// Embedding Comparator Functions
class Comparator {
   public:
    virtual ~Comparator(){};
    virtual torch::Tensor operator()(torch::Tensor src, torch::Tensor dst) = 0;
};

class L2Compare : public Comparator {
   public:
    L2Compare(){};

    torch::Tensor operator()(torch::Tensor src, torch::Tensor dst) override;
};

class CosineCompare : public Comparator {
   public:
    CosineCompare(){};

    torch::Tensor operator()(torch::Tensor src, torch::Tensor dst) override;
};

class DotCompare : public Comparator {
   public:
    DotCompare(){};

    torch::Tensor operator()(torch::Tensor src, torch::Tensor dst) override;
};

#endif  // MARIUS_COMPARATOR_H
