//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_GRAPH_SAGE_LAYER_H
#define MARIUS_GRAPH_SAGE_LAYER_H

#include "gnn_layer.h"

class GraphSageLayer : public GNNLayer {
   public:
    shared_ptr<GraphSageLayerOptions> options_;
    torch::Tensor w1_;
    torch::Tensor w2_;

    GraphSageLayer(shared_ptr<LayerConfig> layer_config, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
};

#endif  // MARIUS_GRAPH_SAGE_LAYER_H
