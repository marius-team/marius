//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_GPU_H
#define MARIUS_PIPELINE_GPU_H

#include "pipeline.h"
#include "queue.h"

class BatchToDeviceWorker : protected Worker {
public:
    BatchToDeviceWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class ComputeWorkerGPU : protected Worker {
public:
    int gpu_id_;

    ComputeWorkerGPU(Pipeline *pipeline, bool *paused, ThreadStatus *status, int gpu_id) : Worker{pipeline, paused, status}, gpu_id_{gpu_id} {}

    void run() override;
};

class EncodeNodesWorkerGPU : protected Worker {
public:
    int gpu_id_;

    EncodeNodesWorkerGPU(Pipeline *pipeline, bool *paused, ThreadStatus *status, int gpu_id) : Worker{pipeline, paused, status}, gpu_id_{gpu_id} {}

    void run() override;
};

class BatchToHostWorker : protected Worker {
public:
    int gpu_id_;

    BatchToHostWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status, int gpu_id) : Worker{pipeline, paused, status}, gpu_id_{gpu_id} {};

    void run() override;
};

class PipelineGPU : public Pipeline {
public:
    vector<std::thread> *pool_[GPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[GPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[GPU_NUM_WORKER_TYPES];

//    shared_ptr<Queue<shared_ptr<Batch>>> loaded_batches_;                     // defined in pipeline.h
    std::vector<shared_ptr<Queue<shared_ptr<Batch>>>> device_loaded_batches_;   // one queue per GPU
    std::vector<shared_ptr<Queue<shared_ptr<Batch>>>> device_update_batches_;   // one queue per GPU
//    shared_ptr<Queue<shared_ptr<Batch>>> update_batches_;                     // defined in pipeline.h

    // these variables should only be accessed/updated when the model->lock is acquired
    std::mutex *gpu_sync_lock_;
    std::condition_variable *gpu_sync_cv_;
    int batches_since_last_sync_;
    int gpu_sync_interval_;

    PipelineGPU(shared_ptr<DataLoader> dataloader,
                shared_ptr<Model> model,
                bool train,
                shared_ptr<ProgressReporter> reporter,
                shared_ptr<PipelineConfig> pipeline_config,
                bool encode_only = false);

    ~PipelineGPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus = 1) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

#endif //MARIUS_PIPELINE_GPU_H
