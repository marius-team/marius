//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GAT_LAYER_H
#define MARIUS_GAT_LAYER_H

#include "layers/gnn_layer.h"

class GATLayer : public GNNLayer, public torch::nn::Cloneable<GATLayer> {
public:
    shared_ptr<GNNLayerConfig> layer_config_;
    shared_ptr<GATLayerOptions> options_;
    int head_dim_;
    float input_dropout_;
    float attention_dropout_;
    torch::Tensor weight_matrices_;
    torch::Tensor a_l_;
    torch::Tensor a_r_;
    torch::Tensor bias_;
    torch::Device device_;

    GATLayer(shared_ptr<GNNLayerConfig> layer_config, torch::Device device);

    void reset() override;

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true) override;
};

#endif //MARIUS_GAT_LAYER_H
