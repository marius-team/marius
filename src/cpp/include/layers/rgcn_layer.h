//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_RGCN_LAYER_H
#define MARIUS_RGCN_LAYER_H

#include "layers/gnn_layer.h"

class RGCNLayer : public GNNLayer, public torch::nn::Cloneable<RGCNLayer> {
public:
    shared_ptr<GNNLayerConfig> layer_config_;
    shared_ptr<GNNLayerOptions> options_;
    int num_relations_;
    bool use_incoming_;
    bool use_outgoing_;
    torch::Tensor relation_matrices_;
    torch::Tensor inverse_relation_matrices_;
    torch::Tensor self_matrix_;
    torch::Tensor bias_;
    torch::Device device_;

    RGCNLayer(shared_ptr<GNNLayerConfig> layer_config, int num_relations, bool use_incoming, bool use_outgoing, torch::Device device);

    void reset() override;

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true) override;

};

#endif //MARIUS_RGCN_LAYER_H
