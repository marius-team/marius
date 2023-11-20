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

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

struct ProcessGroupState {
    shared_ptr<c10d::ProcessGroupGloo> pg;
    string address;
};

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
    shared_ptr<ModelConfig> model_config_;

    bool has_embeddings_;
    bool has_partition_embeddings_;

    shared_ptr<ProcessGroupState> pg_gloo_;
    shared_ptr<DistributedConfig> dist_config_;
    shared_ptr<c10d::ProcessGroupGloo> compute_pg_;
    shared_ptr<c10d::ProcessGroupNCCL> compute_pg_nccl_;
    shared_ptr<c10d::ProcessGroupGloo> total_compute_pg_;
    vector<vector<int>> feeders_;
    vector<int> compute_workers_;
    vector<int> all_workers_;
    vector<int> children_;
    vector<int> parents_;
    bool dist_;
    bool batch_worker_;
    bool compute_worker_;
    bool first_epoch_;
    std::atomic<bool> epoch_complete_;
    std::mutex *pg_lock_;
    std::mutex *update_feeders_lock_;
    int last_compute_worker_;
    std::atomic<bool> already_notified_;

    Model(shared_ptr<GeneralEncoder> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss,
          shared_ptr<Reporter> reporter = nullptr, LearningTask learning_task = LearningTask::LINK_PREDICTION,
          std::vector<shared_ptr<Optimizer>> optimizers_ = {});

    torch::Tensor forward_nc(at::optional<torch::Tensor> node_embeddings, at::optional<torch::Tensor> node_features, DENSEGraph dense_graph, bool train);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_lp(shared_ptr<Batch> batch, bool train);

    void train_batch(shared_ptr<Batch> batch, bool call_step = true);

    void evaluate_batch(shared_ptr<Batch> batch, bool add_result_to_batch = false);

    void clear_grad();

    void clear_grad_all();

    void step();

    void step_all();

    void save(string directory);

    void load(string directory, bool train);

    void createDeviceModels(std::vector<torch::Device> devices);

    void all_reduce();

    void setup_optimizers(shared_ptr<ModelConfig> model_config);

    int64_t get_base_embedding_dim();

    bool has_embeddings();

    bool has_partition_embeddings();

    void setDistPG(shared_ptr<ProcessGroupState> pg_gloo, shared_ptr<DistributedConfig> dist_config) {
        pg_gloo_ = pg_gloo;
        dist_config_ = dist_config;
        dist_ = true;

        if (dist_config_ == nullptr) throw MariusRuntimeException("Expected distributed config to be defined.");

        int ii = 0;
        vector<int> parents;
        for (auto worker_config : dist_config_->workers) {
            if (worker_config->type == WorkerType::BATCH) {
                vector<int> children = std::dynamic_pointer_cast<BatchWorkerOptions>(worker_config->options)->children;
                if (ii == pg_gloo_->pg->getRank()) {
                    children_ = children;
                }
                for (auto c : children) {
                    if (c == pg_gloo_->pg->getRank()) {
                        parents.push_back(ii);
                    }
                }
            }
            ii++;
        }

        parents_ = parents;
    }

    void distGradSync();

    void distModelSync(bool global_pg = false, bool bypass_check = false, bool all_reduce = true, bool optimizers = true, int from_worker = 0);

    void createComputePG(vector<vector<int>> feeders, vector<int> global_to_local, vector<int> local_to_global);

    void distPrepareForTraining(bool eval = false);

    void updateFeeders(int x, bool eval = false);

    void distListenForComplete(bool eval = false);

    void distModelAverage();

    void distNotifyCompleteAndWait(bool eval = false, bool wait = true);
};

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, int num_partitions, bool train,
                                      bool compute_worker = true);

#endif  // MARIUS_INCLUDE_MODEL_H_
