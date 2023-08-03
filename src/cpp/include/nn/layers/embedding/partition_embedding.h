//
// Created by Roger Waleffe on 8/2/23.
//

#ifndef MARIUS_PARTITION_EMBEDDING_H
#define MARIUS_PARTITION_EMBEDDING_H

#include "common/datatypes.h"
#include "nn/layers/layer.h"
#include "storage/storage.h"

class PartitionEmbeddingLayer : public Layer {
public:
//    int offset_;
//    shared_ptr<GraphSageLayerOptions> options_;
    int num_partitions_;
    bool add_to_gnn_input_;
    torch::Tensor emb_table_;

    PartitionEmbeddingLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int num_partitions);

    void reset() override;

    torch::Tensor forward(torch::Tensor input, DENSEGraph dense_graph);

//    torch::Tensor init_embeddings(int64_t num_partitions);
};

#endif //MARIUS_PARTITION_EMBEDDING_H
