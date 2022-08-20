//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_GNN_H
#define MARIUS_GNN_H

#include "layers/gnn_layer.h"
#include "configuration/config.h"

class GeneralGNN : public torch::nn::Cloneable<GeneralGNN> {
public:
    shared_ptr<EncoderConfig> encoder_config_;
    int num_relations_;
    torch::Device device_;

    std::vector<std::shared_ptr<GNNLayer>> layers_;

    GeneralGNN(shared_ptr<EncoderConfig> encoder_config, torch::Device device, int num_relations = 1);

    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train = true);

    void reset() override;

};


#endif //MARIUS_GNN_H
