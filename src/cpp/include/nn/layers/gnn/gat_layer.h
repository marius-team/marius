//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GAT_LAYER_H
#define MARIUS_GAT_LAYER_H

#include "gnn_layer.h"

class GATLayer : public GNNLayer {
   public:
    shared_ptr<GATLayerOptions> options_;
    int head_dim_;
    float input_dropout_;
    float attention_dropout_;
    torch::Tensor weight_matrices_;
    torch::Tensor a_l_;
    torch::Tensor a_r_;

    GATLayer(shared_ptr<LayerConfig> layer_config, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
};

#endif  // MARIUS_GAT_LAYER_H
