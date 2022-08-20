//
// Created by Jason Mohoney on 2/11/21.
//

#ifndef MARIUS_INCLUDE_MODEL_H_
#define MARIUS_INCLUDE_MODEL_H_

#include "configuration/config.h"
#include "decoders/decoder.h"
#include "decoders/complex.h"
#include "decoders/distmult.h"
#include "decoders/transe.h"
#include "encoders/gnn.h"
#include "featurizers/featurizer.h"
#include "loss.h"
#include "regularizer.h"
#include "reporting.h"

using std::shared_ptr;

class Model : public torch::nn::Module {
  public:
    shared_ptr<Featurizer> featurizer_;
    shared_ptr<torch::optim::Optimizer> featurizer_optimizer_;
    shared_ptr<GeneralGNN> encoder_;
    shared_ptr<torch::optim::Optimizer> encoder_optimizer_;
    shared_ptr<Decoder> decoder_;
    shared_ptr<torch::optim::Optimizer> decoder_optimizer_;
    shared_ptr<LossFunction> loss_function_;
    shared_ptr<Regularizer> regularizer_;
    shared_ptr<Reporter> reporter_;

    // TODO set all these properly
    torch::Device current_device_;
    std::vector<torch::Device> devices_;
    LearningTask learning_task_;
    shared_ptr<ModelConfig> model_config_;
    bool has_embeddings_;
    bool has_features_;
    bool reinitialize_;
    bool train_;
    bool filtered_eval_;

    // Multi-GPU training
    std::vector<std::shared_ptr<Model>> device_models_;

    Model();

    Model(shared_ptr<ModelConfig> model_config, torch::Device device);

    virtual void train_batch(Batch *batch) {};

    virtual void train_batch(std::vector<Batch *> sub_batches) {};

    virtual void evaluate(Batch *batch, bool filtered_eval) {};

    virtual void evaluate(std::vector<Batch *> sub_batches, bool filtered_eval) {};

    void zero_grad() override;

    void step();

    void save(string directory);

    void load(string directory);

    std::shared_ptr<Model> clone_to_device(torch::Device device);

    void broadcast(std::vector<torch::Device> devices);

    void allReduce();
};

class NodeClassificationModel : public Model {
  public:
    NodeClassificationModel(shared_ptr<ModelConfig> model_config, shared_ptr<GeneralGNN> encoder, shared_ptr<LossFunction> loss, shared_ptr<Regularizer> regularizer, shared_ptr<Featurizer> featurizer, shared_ptr<Reporter> reporter = nullptr);

    Labels forward(Batch *batch, bool train);

    void train_batch(Batch *batch) override;

    void train_batch(std::vector<Batch *> sub_batches) override;

    void evaluate(Batch *batch, bool filtered_eval = false) override;

    void evaluate(std::vector<Batch *> sub_batches, bool filtered_eval = false) override;
};

class LinkPredictionModel : public Model {
  public:
    LinkPredictionModel(shared_ptr<ModelConfig> model_config, shared_ptr<GeneralGNN> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Regularizer> regularizer, shared_ptr<Featurizer> featurizer, shared_ptr<Reporter> reporter = nullptr);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(Batch *batch, bool train);

    void train_batch(Batch *batch) override;

    void train_batch(std::vector<Batch *> sub_batches) override;

    void evaluate(Batch *batch, bool filtered_eval) override;

    void evaluate(std::vector<Batch *> sub_batches, bool filtered_eval) override;
};

shared_ptr<Model> initializeModel(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations);

shared_ptr<torch::optim::Optimizer> getOptimizerForModule(shared_ptr<torch::nn::Module> module, shared_ptr<OptimizerConfig> optimizer_config);

#endif //MARIUS_INCLUDE_MODEL_H_
