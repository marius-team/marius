//
// Created by Jason Mohoney on 2/11/21.
//

#ifndef MARIUS_INCLUDE_MODEL_H_
#define MARIUS_INCLUDE_MODEL_H_

#include "configuration/config.h"
#include "data/batch.h"
#include "decoders/decoder.h"
#include "encoders/encoder.h"
#include "loss.h"
#include "optim.h"
#include "reporting/reporting.h"

class Model : public torch::nn::Module {
   public:
    shared_ptr<GeneralEncoder> encoder_;
    shared_ptr<Decoder> decoder_;
    shared_ptr<LossFunction> loss_function_;
    shared_ptr<Reporter> reporter_;
    std::vector<shared_ptr<Optimizer>> optimizers_;

    torch::Device device_;
    LearningTask learning_task_;
    float sparse_lr_;

    // Multi-GPU training
    std::vector<shared_ptr<Model>> device_models_;

    Model(shared_ptr<GeneralEncoder> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Reporter> reporter = nullptr,
          std::vector<shared_ptr<Optimizer>> optimizers_ = {});

    torch::Tensor forward_nc(at::optional<torch::Tensor> node_embeddings, at::optional<torch::Tensor> node_features, DENSEGraph dense_graph, bool train);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_lp(shared_ptr<Batch> batch, bool train);

    void train_batch(shared_ptr<Batch> batch, bool call_step = true);

    void evaluate_batch(shared_ptr<Batch> batch);

    void clear_grad();

    void clear_grad_all();

    void step();

    void step_all();

    void save(string directory);

    void load(string directory, bool train);

    void broadcast(std::vector<torch::Device> devices);

    void all_reduce();

    void setup_optimizers(shared_ptr<ModelConfig> model_config);

    int64_t get_base_embedding_dim();

    bool has_embeddings();
};

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, bool train);

#endif  // MARIUS_INCLUDE_MODEL_H_
