//
// Created by Jason Mohoney on 2/1/22.
//

#ifndef MARIUS_EMBEDDING_H
#define MARIUS_EMBEDDING_H

#include "common/datatypes.h"
#include "nn/layers/layer.h"
#include "storage/storage.h"

class EmbeddingLayer : public Layer {
   public:
    int offset_;

    EmbeddingLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int offset = 0);

    torch::Tensor forward(torch::Tensor input);

    torch::Tensor init_embeddings(int64_t num_nodes);

    void reset() override;
};

#endif  // MARIUS_EMBEDDING_H
