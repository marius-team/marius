//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_CPU_H
#define MARIUS_PIPELINE_CPU_H

#include "pipeline.h"
#include "queue.h"

class ComputeWorkerCPU : protected Worker {
public:
    ComputeWorkerCPU(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {}

    void run() override;
};

class EncodeNodesWorkerCPU : protected Worker {
public:
    int gpu_id_;

    EncodeNodesWorkerCPU(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {}

    void run() override;
};

class AccumulateGradientsWorker : protected Worker {
public:
    AccumulateGradientsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class PipelineCPU : public Pipeline {
public:
    vector<std::thread> *pool_[CPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[CPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[CPU_NUM_WORKER_TYPES];

//    shared_ptr<Queue<shared_ptr<Batch>>> loaded_batches_;
    shared_ptr<Queue<shared_ptr<Batch>>> unaccumulated_batches_;
//    shared_ptr<Queue<shared_ptr<Batch>>> update_batches_;

    PipelineCPU(shared_ptr<DataLoader> dataloader,
                shared_ptr<Model> model,
                bool train,
                shared_ptr<ProgressReporter> reporter,
                shared_ptr<PipelineConfig> pipeline_config,
                bool encode_only = false);

    ~PipelineCPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus = 1) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

#endif //MARIUS_PIPELINE_CPU_H
