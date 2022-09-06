//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_RGCN_LAYER_H
#define MARIUS_RGCN_LAYER_H

#include "gnn_layer.h"

class RGCNLayer : public GNNLayer {
   public:
    shared_ptr<GNNLayerOptions> options_;
    int num_relations_;
    torch::Tensor relation_matrices_;
    torch::Tensor inverse_relation_matrices_;
    torch::Tensor self_matrix_;

    RGCNLayer(shared_ptr<LayerConfig> layer_config, int num_relations, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
};

#endif  // MARIUS_RGCN_LAYER_H
