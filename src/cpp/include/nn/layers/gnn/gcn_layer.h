//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GCN_LAYER_H
#define MARIUS_GCN_LAYER_H

#include "gnn_layer.h"

class GCNLayer : public GNNLayer {
   public:
    shared_ptr<GNNLayerOptions> options_;
    bool use_incoming_;
    bool use_outgoing_;
    torch::Tensor w_;

    GCNLayer(shared_ptr<LayerConfig> layer_config, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
};

#endif  // MARIUS_GCN_LAYER_H
