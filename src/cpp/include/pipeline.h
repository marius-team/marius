//
// Created by Jason Mohoney on 2/29/20.
//
#ifndef MARIUS_PIPELINE_H
#define MARIUS_PIPELINE_H

#include <time.h>

#include "batch.h"
#include "dataloader.h"
#include "datatypes.h"
#include "model.h"

#define WAIT_TIME 100000 // 100 micro seconds
#define NANOSECOND 1
#define MICROSECOND 1000
#define MILLISECOND 1000000

// CPU Pipeline worker IDs
#define EMBEDDINGS_LOADER_ID 0
#define CPU_COMPUTE_ID 1
#define CPU_ACCUMULATE_ID 2
#define UPDATE_EMBEDDINGS_ID 3

// GPU Pipeline worker IDs
#define EMBEDDINGS_TRANSFER_ID 4
#define GPU_COMPUTE_ID 5
#define UPDATE_TRANSFER_ID 6

#define CPU_NUM_WORKER_TYPES 4
#define GPU_NUM_WORKER_TYPES 5

using std::atomic;
using std::thread;
using std::get;

enum class ThreadStatus {
    Running,
    WaitPush,
    WaitPop,
    Paused,
    Done
};

template<class T>
class Queue {
  private:
    int max_size_;
  public:
    std::deque<T> queue_;
    std::mutex *mutex_;
    std::condition_variable *cv_;
    std::atomic<bool> expecting_data_;

    Queue<T>(int max_size);

    bool push(T item) {
        bool result = true;
        if (isFull()) {
            result = false;
        } else {
            queue_.push_back(item);
        }
        return result;
    }

    void blocking_push(T item) {
        bool pushed = false;
        while (!pushed) {
            std::unique_lock lock(*mutex_);
            pushed = push(item);
            if (!pushed) {
                cv_->wait(lock);
            } else {
                cv_->notify_all();
            }
            lock.unlock();
        }
    }

    std::tuple<bool, T> pop() {
        bool result = true;
        T item;
        if (isEmpty()) {
            result = false;
        } else {
            item = queue_.front();
            queue_.pop_front();
        }
        return std::forward_as_tuple(result, item);
    }

    std::tuple<bool, T> blocking_pop() {
        bool popped = false;
        T item;
        while (!popped && expecting_data_) {
            std::unique_lock lock(*mutex_);
            auto tup = pop();
            popped = get<0>(tup);
            item = get<1>(tup);
            if (!popped) {
                cv_->wait(lock);
            } else {
                cv_->notify_all();
            }
            lock.unlock();
        }
        return std::forward_as_tuple(popped, item);
    }

    void lock() {
        mutex_->lock();
    }

    void unlock() {
        mutex_->unlock();
    }

    void flush() {
        lock();
        queue_ = std::deque<T>();
        unlock();
    }

    int size() {
        return queue_.size();
    }

    bool isFull() {
        return queue_.size() == max_size_;
    }

    bool isEmpty() {
        return queue_.size() == 0;
    }

    int getMaxSize() {
        return max_size_;
    }

    typedef typename std::deque<T> queue_type;

    typedef typename queue_type::iterator iterator;
    typedef typename queue_type::const_iterator const_iterator;

    inline iterator begin() noexcept { return queue_.begin(); }

    inline const_iterator cbegin() const noexcept { return queue_.cbegin(); }

    inline iterator end() noexcept { return queue_.end(); }

    inline const_iterator cend() const noexcept { return queue_.cend(); }
};

class Pipeline {
  public:
    DataLoader *dataloader_;
    std::shared_ptr<Model> model_;
    ProgressReporter *reporter_;
    shared_ptr<PipelineConfig> pipeline_options_;

    int staleness_bound_;
    atomic<int> batches_in_flight_;
    std::mutex *max_batches_lock_;
    std::condition_variable *max_batches_cv_;
    atomic<int64_t> edges_processed_;

    std::mutex *pipeline_lock_;
    std::condition_variable pipeline_cv_;

    atomic<int> admitted_batches_;

    bool train_;

  public:

    ~Pipeline();

    thread initThreadOfType(int worker_type, bool *paused, ThreadStatus *status);

    virtual void addWorkersToPool(int pool_id, int worker_type, int num_workers) = 0;

    bool isDone();

    bool isTrain();

    bool hasEmbeddings();

    void waitComplete();

    virtual void initialize() = 0;

    virtual void start() = 0;

    virtual void stopAndFlush() = 0;

    virtual void flushQueues() = 0;

    virtual void setQueueExpectingData(bool expecting_data) = 0;

    virtual void reportQueueStatus() {};

    virtual void reportThreadStatus() {};
};

class PipelineCPU : public Pipeline {
  public:
    vector<thread> *pool_[CPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[CPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[CPU_NUM_WORKER_TYPES];

    Queue<Batch *> *loaded_batches_;
    Queue<Batch *> *unaccumulated_batches_;
    Queue<Batch *> *update_batches_;

    PipelineCPU(DataLoader *dataloader_,
                std::shared_ptr<Model> model,
                bool train,
                ProgressReporter *reporter,
                shared_ptr<PipelineConfig> pipeline_config);

    ~PipelineCPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

class PipelineGPU : public Pipeline {
  public:
    vector<thread> *pool_[GPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[GPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[GPU_NUM_WORKER_TYPES];

    Queue<std::vector<Batch *>> *loaded_batches_;
    Queue<std::vector<Batch *>> *device_loaded_batches_;
    Queue<std::vector<Batch *>> *device_update_batches_;
    Queue<std::vector<Batch *>> *update_batches_;

    PipelineGPU(DataLoader *dataloader_,
                std::shared_ptr<Model> model,
                bool train,
                ProgressReporter *reporter,
                shared_ptr<PipelineConfig> pipeline_config);

    ~PipelineGPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;
};

class Worker {
  protected:
    Pipeline *pipeline_;
    struct timespec sleep_time_;
    bool *paused_;
    ThreadStatus *status_;

  public:
    Worker(Pipeline *pipeline, bool *paused, ThreadStatus *status);

    virtual void run() = 0;
};

class LoadEmbeddingsWorker : protected Worker {
  public:
    LoadEmbeddingsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class BatchToDeviceWorker : protected Worker {
  public:
    BatchToDeviceWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class PrepareBatchWorker : protected Worker {
  public:
    PrepareBatchWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class ComputeWorkerCPU : protected Worker {
  public:
    ComputeWorkerCPU(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {}

    void run() override;
};

class ComputeWorkerGPU : protected Worker {
  public:
    ComputeWorkerGPU(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {}

    void run() override;
};

class AccumulateGradientsWorker : protected Worker {
  public:
    AccumulateGradientsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class GradientsToHostWorker : protected Worker {
  public:
    GradientsToHostWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class UpdateEmbeddingsWorker : protected Worker {
  public:
    UpdateEmbeddingsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

#endif //MARIUS_PIPELINE_H


