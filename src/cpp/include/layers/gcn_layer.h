//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GCN_LAYER_H
#define MARIUS_GCN_LAYER_H

#include "layers/gnn_layer.h"

class GCNLayer : public GNNLayer, public torch::nn::Cloneable<GCNLayer> {
public:
    shared_ptr<GNNLayerConfig> layer_config_;
    shared_ptr<GNNLayerOptions> options_;
    bool use_incoming_;
    bool use_outgoing_;
    torch::Tensor w_;
    torch::Tensor bias_;
    torch::Device device_;

    GCNLayer(shared_ptr<GNNLayerConfig> layer_config, bool use_incoming, bool use_outgoing, torch::Device device);

    void reset() override;

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true) override;
};

#endif //MARIUS_GCN_LAYER_H
