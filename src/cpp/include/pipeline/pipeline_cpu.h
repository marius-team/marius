//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_CPU_H
#define MARIUS_PIPELINE_CPU_H

#include "pipeline.h"
#include "queue.h"

class ComputeWorkerCPU : public Worker {
   public:
    ComputeWorkerCPU(Pipeline *pipeline) : Worker{pipeline} {}

    void run() override;
};

class EncodeNodesWorkerCPU : public Worker {
   public:
    int gpu_id_;

    EncodeNodesWorkerCPU(Pipeline *pipeline) : Worker{pipeline} {}

    void run() override;
};

class PipelineCPU : public Pipeline {
   public:
    vector<shared_ptr<Worker>> pool_[CPU_NUM_WORKER_TYPES];

    PipelineCPU(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, bool train, shared_ptr<ProgressReporter> reporter,
                shared_ptr<PipelineConfig> pipeline_config, bool encode_only = false);

    ~PipelineCPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus = 1) override;

    void initialize() override;

    void start() override;

    void pauseAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

#endif  // MARIUS_PIPELINE_CPU_H
