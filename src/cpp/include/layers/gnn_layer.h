//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GNN_LAYER_H
#define MARIUS_GNN_LAYER_H

#include "configuration/config.h"
#include "datatypes.h"
#include "initialization.h"
#include "graph.h"

class GNNLayer {
public:
    int input_dim_;
    int output_dim_;

    virtual ~GNNLayer() {};

    virtual Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {return torch::Tensor();};

};

#endif //MARIUS_GNN_LAYER_H
