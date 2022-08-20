//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_FEATURIZER_H_
#define MARIUS_FEATURIZER_H_

#include "datatypes.h"

/**
  Generates new embeddings for nodes by combining node features and their respective embeddings in order to emphasize individual node properties.
*/
class Featurizer : public torch::nn::Module {
  public:
    virtual ~Featurizer() {};

    /**
      Combines node features with their node embeddings to generate new embeddings.
      @param node_features The node features
      @param node_embeddings The node embeddings
      @return The new embeddings generated from combining input node features and node embeddings
    */
    virtual Embeddings operator()(Features node_features, Embeddings node_embeddings) = 0;
};

class CatFeaturizer : public Featurizer {
  private:
  public:
    CatFeaturizer(int norm, float coefficient);

    Embeddings operator()(Features node_features, Embeddings node_embeddings) override;
};


#endif //MARIUS_FEATURIZER_H_
