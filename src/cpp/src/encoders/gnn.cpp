//
// Created by Jason Mohoney on 9/29/21.
//

#include "encoders/gnn.h"

#include "activation.h"
#include "layers/gcn_layer.h"
#include "layers/gat_layer.h"
#include "layers/rgcn_layer.h"
#include "layers/graph_sage_layer.h"

GeneralGNN::GeneralGNN(shared_ptr<EncoderConfig> encoder_config, torch::Device device, int num_relations) : device_(torch::Device(torch::kCPU)) {
    encoder_config_ = encoder_config;
    num_relations_ = num_relations;
    device_ = device;

    reset();
}

void GeneralGNN::reset() {
    layers_.clear();

    std::shared_ptr<GNNLayer> layer;
    int layer_id = 0;
    for (auto layer_config : encoder_config_->layers) {
        if (layer_config->type == GNNLayerType::GRAPH_SAGE) {
            layer = std::make_shared<GraphSageLayer>(layer_config, encoder_config_->use_incoming_nbrs, encoder_config_->use_outgoing_nbrs, device_);
            register_module<GraphSageLayer>("layer:" + std::to_string(layer_id++), std::dynamic_pointer_cast<GraphSageLayer>(layer));
        } else if (layer_config->type == GNNLayerType::GAT) {
            layer = std::make_shared<GATLayer>(layer_config, device_);
            register_module<GATLayer>("layer:" + std::to_string(layer_id++), std::dynamic_pointer_cast<GATLayer>(layer));
        } else if (layer_config->type == GNNLayerType::GCN) {
            layer = std::make_shared<GCNLayer>(layer_config, encoder_config_->use_incoming_nbrs, encoder_config_->use_outgoing_nbrs,device_);
            register_module<GCNLayer>("layer:" + std::to_string(layer_id++), std::dynamic_pointer_cast<GCNLayer>(layer));
        } else if (layer_config->type == GNNLayerType::RGCN) {
            layer = std::make_shared<RGCNLayer>(layer_config, num_relations_, encoder_config_->use_incoming_nbrs, encoder_config_->use_outgoing_nbrs, device_);
            register_module<RGCNLayer>("layer:" + std::to_string(layer_id++), std::dynamic_pointer_cast<RGCNLayer>(layer));
        } else {
            throw std::runtime_error("Unimplemented GNNLayer");
        }
        layers_.push_back(layer);
    }
}

Embeddings GeneralGNN::forward(Embeddings inputs, GNNGraph gnn_graph, bool train) {

    Embeddings outputs = inputs;

    for (int i = 0; i < layers_.size(); i++) {
        outputs = layers_[i]->forward(outputs, gnn_graph, train);
        outputs = apply_activation(encoder_config_->layers[i]->activation, outputs);
        if (i < layers_.size() - 1) {
            gnn_graph.prepareForNextLayer();
        }
    }

    return outputs;
}
