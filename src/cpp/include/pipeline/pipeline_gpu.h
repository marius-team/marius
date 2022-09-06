//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_GPU_H
#define MARIUS_PIPELINE_GPU_H

#include "pipeline.h"
#include "queue.h"

class BatchToDeviceWorker : public Worker {
   public:
    BatchToDeviceWorker(Pipeline *pipeline) : Worker{pipeline} {};

    void run() override;
};

class ComputeWorkerGPU : public Worker {
   public:
    int gpu_id_;

    ComputeWorkerGPU(Pipeline *pipeline, int gpu_id) : Worker{pipeline}, gpu_id_{gpu_id} {}

    void run() override;
};

class EncodeNodesWorkerGPU : public Worker {
   public:
    int gpu_id_;

    EncodeNodesWorkerGPU(Pipeline *pipeline, int gpu_id) : Worker{pipeline}, gpu_id_{gpu_id} {}

    void run() override;
};

class BatchToHostWorker : public Worker {
   public:
    int gpu_id_;

    BatchToHostWorker(Pipeline *pipeline, int gpu_id) : Worker{pipeline}, gpu_id_{gpu_id} {};

    void run() override;
};

class PipelineGPU : public Pipeline {
   public:
    vector<shared_ptr<Worker>> pool_[GPU_NUM_WORKER_TYPES];

    std::vector<shared_ptr<Queue<shared_ptr<Batch>>>> device_loaded_batches_;  // one queue per GPU
    std::vector<shared_ptr<Queue<shared_ptr<Batch>>>> device_update_batches_;  // one queue per GPU

    // these variables should only be accessed/updated when the model->lock is acquired
    std::mutex *gpu_sync_lock_;
    std::condition_variable *gpu_sync_cv_;
    int batches_since_last_sync_;
    int gpu_sync_interval_;

    PipelineGPU(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, bool train, shared_ptr<ProgressReporter> reporter,
                shared_ptr<PipelineConfig> pipeline_config, bool encode_only = false);

    ~PipelineGPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus = 1) override;

    void initialize() override;

    void start() override;

    void pauseAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

#endif  // MARIUS_PIPELINE_GPU_H
