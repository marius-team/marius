//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_ENCODER_H
#define MARIUS_ENCODER_H

#include "configuration/config.h"
#include "nn/layers/layer.h"

class GeneralEncoder : public torch::nn::Cloneable<GeneralEncoder> {
   public:
    shared_ptr<EncoderConfig> encoder_config_;
    int num_relations_;
    torch::Device device_;
    bool has_features_;
    bool has_embeddings_;

    std::vector<std::vector<shared_ptr<Layer>>> layers_;

    GeneralEncoder(shared_ptr<EncoderConfig> encoder_config, torch::Device device, int num_relations = 1);

    GeneralEncoder(std::vector<std::vector<shared_ptr<Layer>>> layers);

    torch::Tensor forward(at::optional<torch::Tensor> embeddings, at::optional<torch::Tensor> features, DENSEGraph dense_graph, bool train = true);

    void reset() override;

    std::shared_ptr<Layer> initEmbeddingLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id);

    std::shared_ptr<Layer> initFeatureLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id);

    std::shared_ptr<Layer> initDenseLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id);

    std::shared_ptr<Layer> initGNNLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id, int sampling_id);

    std::shared_ptr<Layer> initReductionLayer(shared_ptr<LayerConfig> layer_config, int stage_id, int layer_id);
};

#endif  // MARIUS_ENCODER_H
