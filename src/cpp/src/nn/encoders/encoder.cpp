//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/encoders/encoder.h"

#include "nn/activation.h"
#include "nn/layers/embedding/embedding.h"
#include "nn/layers/feature/feature.h"
#include "nn/layers/gnn/gat_layer.h"
#include "nn/layers/gnn/gcn_layer.h"
#include "nn/layers/gnn/graph_sage_layer.h"
#include "nn/layers/gnn/rgcn_layer.h"
#include "nn/layers/reduction/concat.h"
#include "nn/layers/reduction/linear.h"
#include "nn/layers/reduction/reduction_layer.h"

GeneralEncoder::GeneralEncoder(shared_ptr<EncoderConfig> encoder_config, torch::Device device, int num_relations) : device_(torch::kCPU) {
    encoder_config_ = encoder_config;
    num_relations_ = num_relations;
    device_ = device;

    has_features_ = false;
    has_embeddings_ = false;

    reset();
}

GeneralEncoder::GeneralEncoder(std::vector<std::vector<shared_ptr<Layer>>> layers) : device_(torch::kCPU) {
    layers_ = layers;
    device_ = layers_[0][0]->device_;

    int stage_id = 0;
    for (auto stage : layers_) {
        int layer_id = 0;
        for (auto layer : stage) {
            if (layer->device_ != device_) {
                throw MariusRuntimeException("All layers of the encoder must use the same device.");
            }

            // TODO unify with initLayer functions
            string name;
            if (instance_of<Layer, EmbeddingLayer>(layer)) {
                name = "embedding:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                register_module<EmbeddingLayer>(name, std::dynamic_pointer_cast<EmbeddingLayer>(layer));
            } else if (instance_of<Layer, FeatureLayer>(layer)) {
                name = "feature:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                register_module<FeatureLayer>(name, std::dynamic_pointer_cast<FeatureLayer>(layer));
            } else if (instance_of<Layer, ReductionLayer>(layer)) {
                if (instance_of<Layer, LinearReduction>(layer)) {
                    name = "linear_reduction:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<LinearReduction>(name, std::dynamic_pointer_cast<LinearReduction>(layer));
                } else if (instance_of<Layer, ConcatReduction>(layer)) {
                    name = "concat_reduction:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<ConcatReduction>(name, std::dynamic_pointer_cast<ConcatReduction>(layer));
                } else {
                    throw std::runtime_error("Unrecognized reduction layer type");
                }
            } else if (instance_of<Layer, GNNLayer>(layer)) {
                if (instance_of<Layer, GraphSageLayer>(layer)) {
                    string name = "graph_sage_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<GraphSageLayer>(name, std::dynamic_pointer_cast<GraphSageLayer>(layer));
                } else if (instance_of<Layer, GATLayer>(layer)) {
                    string name = "gat_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<GATLayer>(name, std::dynamic_pointer_cast<GATLayer>(layer));
                } else if (instance_of<Layer, GCNLayer>(layer)) {
                    string name = "gcn_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<GCNLayer>(name, std::dynamic_pointer_cast<GCNLayer>(layer));
                } else if (instance_of<Layer, RGCNLayer>(layer)) {
                    string name = "rgcn_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
                    register_module<RGCNLayer>(name, std::dynamic_pointer_cast<RGCNLayer>(layer));
                } else {
                    throw std::runtime_error("Unrecognized GNN layer type");
                }
            } else {
                throw std::runtime_error("Unsupported layer type");
            }
            layer_id++;
        }
        stage_id++;
    }
    encoder_config_ = nullptr;
}

shared_ptr<Layer> GeneralEncoder::initEmbeddingLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id) {
    string name = "embedding:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
    shared_ptr<Layer> layer = std::make_shared<EmbeddingLayer>(layer_config, device_);
    register_module<EmbeddingLayer>(name, std::dynamic_pointer_cast<EmbeddingLayer>(layer));
    has_embeddings_ = true;
    return layer;
}

shared_ptr<Layer> GeneralEncoder::initFeatureLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id) {
    string name = "feature:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
    shared_ptr<Layer> layer = std::make_shared<FeatureLayer>(layer_config, device_);
    register_module<FeatureLayer>(name, std::dynamic_pointer_cast<FeatureLayer>(layer));
    has_features_ = true;
    return layer;
}

shared_ptr<Layer> GeneralEncoder::initReductionLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id) {
    auto options = std::dynamic_pointer_cast<ReductionLayerOptions>(layer_config->options);

    if (options->type == ReductionLayerType::LINEAR) {
        string name = "linear_reduction:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        shared_ptr<Layer> layer = std::make_shared<LinearReduction>(layer_config, device_);
        register_module<LinearReduction>(name, std::dynamic_pointer_cast<LinearReduction>(layer));
        return layer;
    } else if (options->type == ReductionLayerType::CONCAT) {
        string name = "concat_reduction:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        shared_ptr<Layer> layer = std::make_shared<ConcatReduction>(layer_config, device_);
        register_module<ConcatReduction>(name, std::dynamic_pointer_cast<ConcatReduction>(layer));
        return layer;
    } else {
        throw std::runtime_error("Unrecognized reduction layer type");
    }
}

shared_ptr<Layer> GeneralEncoder::initGNNLayer(std::shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id, int sampling_id) {
    auto options = std::dynamic_pointer_cast<GNNLayerOptions>(layer_config->options);

    std::shared_ptr<Layer> layer;

    if (options->type == GNNLayerType::GRAPH_SAGE) {
        string name = "graph_sage_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        layer = std::make_shared<GraphSageLayer>(layer_config, device_);
        register_module<GraphSageLayer>(name, std::dynamic_pointer_cast<GraphSageLayer>(layer));
    } else if (options->type == GNNLayerType::GAT) {
        string name = "gat_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        layer = std::make_shared<GATLayer>(layer_config, device_);
        register_module<GATLayer>(name, std::dynamic_pointer_cast<GATLayer>(layer));
    } else if (options->type == GNNLayerType::GCN) {
        string name = "gcn_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        layer = std::make_shared<GCNLayer>(layer_config, device_);
        register_module<GCNLayer>(name, std::dynamic_pointer_cast<GCNLayer>(layer));
    } else if (options->type == GNNLayerType::RGCN) {
        string name = "rgcn_layer:" + std::to_string(stage_id) + "_" + std::to_string(layer_id);
        layer = std::make_shared<RGCNLayer>(layer_config, num_relations_, device_);
        register_module<RGCNLayer>(name, std::dynamic_pointer_cast<RGCNLayer>(layer));
    } else {
        throw std::runtime_error("Unrecognized GNN layer type");
    }

    return layer;
}

void GeneralEncoder::reset() {
    if (encoder_config_ != nullptr) {
        layers_.clear();

        int num_sampling_layers = encoder_config_->train_neighbor_sampling.size();

        if (num_sampling_layers == 0) {
            num_sampling_layers = encoder_config_->eval_neighbor_sampling.size();
        }
        int curr_sampling_layer = 0;

        int stage_id = 0;
        for (auto stage_config : encoder_config_->layers) {
            std::vector<std::shared_ptr<Layer>> stage_layer;

            int layer_id = 0;
            for (auto layer_config : stage_config) {
                std::shared_ptr<Layer> layer;

                if (layer_config->type == LayerType::EMBEDDING) {
                    layer = initEmbeddingLayer(layer_config, stage_id, layer_id);
                } else if (layer_config->type == LayerType::FEATURE) {
                    layer = initFeatureLayer(layer_config, stage_id, layer_id);
                } else if (layer_config->type == LayerType::REDUCTION) {
                    layer = initReductionLayer(layer_config, stage_id, layer_id);
                } else if (layer_config->type == LayerType::GNN) {
                    assert(curr_sampling_layer < num_sampling_layers);
                    layer = initGNNLayer(layer_config, stage_id, layer_id, curr_sampling_layer);
                    curr_sampling_layer++;
                } else {
                    throw std::runtime_error("Unsupported layer type");
                }

                stage_layer.push_back(layer);
                layer_id++;
            }
            layers_.push_back(stage_layer);
            stage_id++;
        }
    } else {
        for (auto stage : layers_) {
            for (auto layer : stage) {
                layer->reset();
            }
        }
    }
}

torch::Tensor GeneralEncoder::forward(at::optional<torch::Tensor> embeddings, at::optional<torch::Tensor> features, DENSEGraph dense_graph, bool train) {
    dense_graph.performMap();

    std::vector<torch::Tensor> outputs = {};

    for (int i = 0; i < layers_.size(); i++) {
        bool use_sample = false;
        bool added_output = false;

        int64_t output_size;
        if (embeddings.has_value() && embeddings.value().defined()) {
            output_size = embeddings.value().size(0);
        } else if (features.has_value() && features.value().defined()) {
            output_size = features.value().size(0);
        } else {
            throw MariusRuntimeException("Encoder requires embeddings and/or features as input");
        }

        for (int j = 0; j < layers_[i].size(); j++) {
            if (instance_of<Layer, GNNLayer>(layers_[i][j])) {
                output_size = dense_graph.node_ids_.size(0) - (dense_graph.hop_offsets_[1].item<int64_t>() - dense_graph.hop_offsets_[0].item<int64_t>());
            }
        }

        std::vector<torch::Tensor> max_outputs(layers_[i].size());
        for (int j = 0; j < layers_[i].size(); j++) {
            if (instance_of<Layer, EmbeddingLayer>(layers_[i][j])) {
                max_outputs[j] = std::dynamic_pointer_cast<EmbeddingLayer>(layers_[i][j])->forward(embeddings.value().narrow(0, 0, output_size));
                max_outputs[j] = layers_[i][j]->post_hook(max_outputs[j]);
                added_output = true;
            } else if (instance_of<Layer, FeatureLayer>(layers_[i][j])) {
                max_outputs[j] = std::dynamic_pointer_cast<FeatureLayer>(layers_[i][j])->forward(features.value().narrow(0, 0, output_size));
                max_outputs[j] = layers_[i][j]->post_hook(max_outputs[j]);
                added_output = true;
            } else if (instance_of<Layer, ReductionLayer>(layers_[i][j])) {
                std::vector<torch::Tensor> new_outputs(1);
                new_outputs[0] = std::dynamic_pointer_cast<ReductionLayer>(layers_[i][j])->forward(outputs);
                new_outputs[0] = layers_[i][j]->post_hook(new_outputs[0]);
                outputs = new_outputs;
            } else if (instance_of<Layer, GNNLayer>(layers_[i][j])) {
                outputs[j] = std::dynamic_pointer_cast<GNNLayer>(layers_[i][j])->forward(outputs[j], dense_graph, train);
                outputs[j] = layers_[i][j]->post_hook(outputs[j]);
                use_sample = true;
            } else {
                throw std::runtime_error("Unsupported layer type");
            }
        }
        // added embedding / features in this stage
        if (added_output) {
            for (int j = outputs.size(); j < max_outputs.size(); j++) {
                outputs.emplace_back(max_outputs[j]);
            }
        }

        // used GNN layer at this stage
        if (use_sample && i < layers_.size() - 1) {
            dense_graph.prepareForNextLayer();
        }
    }

    assert(outputs.size() == 1);

    return outputs[0];
}