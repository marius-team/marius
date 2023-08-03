//
// Created by Roger Waleffe on 8/2/23.
//

#include "nn/layers/embedding/partition_embedding.h"

#include "nn/initialization.h"

PartitionEmbeddingLayer::PartitionEmbeddingLayer(shared_ptr<LayerConfig> layer_config, torch::Device device, int num_partitions) {
    config_ = layer_config;
//    offset_ = offset;
    num_partitions_ = num_partitions;
    device_ = device;

    add_to_gnn_input_ = std::dynamic_pointer_cast<PartitionEmbeddingLayerOptions>(config_->options)->add_to_gnn_input;

    reset();
}

void PartitionEmbeddingLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    emb_table_ = initialize_tensor(config_->init, {num_partitions_, config_->output_dim}, tensor_options).set_requires_grad(true);
    emb_table_ = register_parameter("emb_table", emb_table_);

    if (config_->bias) {
        init_bias();
    }
}

torch::Tensor PartitionEmbeddingLayer::forward(torch::Tensor input, DENSEGraph dense_graph) {
    torch::Tensor node_ids;

    if (add_to_gnn_input_ or !dense_graph.hop_offsets_.defined()){
        node_ids = dense_graph.node_ids_;
    } else {
        int64_t num_nodes_to_remove = (dense_graph.hop_offsets_[1] - dense_graph.hop_offsets_[0]).item<int64_t>();
        node_ids = dense_graph.node_ids_.narrow(0, num_nodes_to_remove, dense_graph.node_ids_.size(0) - num_nodes_to_remove);
    }

    node_ids = node_ids.divide(dense_graph.partition_size_, "trunc");
    if (dense_graph.buffer_state_.defined()) {
        // convert to global partition number if buffer_state_ is defined, otherwise we already have global number (e.g., not using in memory subgraph)
        node_ids = dense_graph.buffer_state_.index_select(0, node_ids);
    }

    input = input + emb_table_.index_select(0, node_ids);

    return input;
}