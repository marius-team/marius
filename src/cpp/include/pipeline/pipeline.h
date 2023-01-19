//
// Created by Jason Mohoney on 2/29/20.
//
#ifndef MARIUS_PIPELINE_H
#define MARIUS_PIPELINE_H

#include <time.h>

#include "common/datatypes.h"
#include "data/batch.h"
#include "data/dataloader.h"
#include "nn/model.h"
#include "pipeline_constants.h"
#include "queue.h"

class Pipeline;

class Worker {
   protected:
    Pipeline *pipeline_;
    struct timespec sleep_time_;
    std::atomic<bool> paused_;
    std::atomic<bool> done_;

    std::thread thread_;

   public:
    explicit Worker(Pipeline *pipeline);

    virtual void run() = 0;

    void spawn() { thread_ = std::thread(&Worker::run, this); }

    void start() { paused_ = false; }

    void pause() { paused_ = true; }

    void stop() {
        paused_ = true;
        done_ = true;

        if (thread_.joinable()) {
            thread_.join();
        }
    }
};

class LoadBatchWorker : public Worker {
   public:
    int worker_id_;

    LoadBatchWorker(Pipeline *pipeline, int worker_id) : Worker{pipeline}, worker_id_{worker_id} {};

    void run() override;
};

class UpdateBatchWorker : public Worker {
   public:
    UpdateBatchWorker(Pipeline *pipeline) : Worker{pipeline} {};

    void run() override;
};

class WriteNodesWorker : public Worker {
   public:
    WriteNodesWorker(Pipeline *pipeline) : Worker{pipeline} {}

    void run() override;
};

class Pipeline {
   public:
    shared_ptr<DataLoader> dataloader_;
    shared_ptr<Model> model_;
    shared_ptr<ProgressReporter> reporter_;
    shared_ptr<PipelineConfig> pipeline_options_;

    int staleness_bound_;
    std::atomic<int> batches_in_flight_;
    std::mutex *max_batches_lock_;
    std::condition_variable *max_batches_cv_;
    std::atomic<int64_t> edges_processed_;

    shared_ptr<Queue<shared_ptr<Batch>>> loaded_batches_;
    shared_ptr<Queue<shared_ptr<Batch>>> update_batches_;

    std::mutex *pipeline_lock_;
    std::condition_variable pipeline_cv_;

    std::atomic<int> admitted_batches_;

    std::atomic<int> assign_id_;

    bool encode_only_;
    bool train_;

    int64_t curr_pos_;

    ~Pipeline();

    shared_ptr<Worker> initWorkerOfType(int worker_type, int gpu_id = 0, int worker_id = 0);

    virtual void addWorkersToPool(int pool_id, int worker_type, int num_workers, int num_gpus = 1) = 0;

    bool isDone();

    bool isTrain();

    bool has_embeddings();

    void waitComplete();

    virtual void initialize() = 0;

    virtual void start() = 0;

    virtual void pauseAndFlush() = 0;

    virtual void flushQueues() = 0;

    virtual void setQueueExpectingData(bool expecting_data) = 0;

    virtual void reportQueueStatus(){};

    virtual void reportThreadStatus(){};
};

#endif  // MARIUS_PIPELINE_H
