//
// Created by Jason Mohoney on 5/26/20.
//

#ifndef MARIUS_BUFFER_H
#define MARIUS_BUFFER_H

#include "common/datatypes.h"
#include "data/batch.h"

class Partition {
   public:
    std::mutex *lock_;            /**< Mutex lock to prevent race conditions */
    std::condition_variable *cv_; /**< Condition variable for signaling */
    void *data_ptr_;              /**< Pointer to partition in memory */
    int partition_id_;            /**< ID of the partition */

    bool present_; /**< If true this partition is present in the buffer */

    int64_t partition_size_; /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;     /**< Number of elements in each embedding */
    torch::Dtype dtype_;     /**< Datatype of the embeddings */
    int dtype_size_;         /**< Size in bytes of the datatype */
    int64_t total_size_;     /**< Total size in bytes of the partition */

    int64_t idx_offset_;  /**< Embedding ID offset of the partition */
    int64_t file_offset_; /**< Offset in bytes of the partition in the embedding file */
    int buffer_idx_;      /**< Buffer entry ID of the partition in the buffer */

    torch::Tensor tensor_; /**< Tensor view of the partition */

    bool evicting_;

    Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset);

    ~Partition();

    torch::Tensor indexRead(Indices indices);
};

class PartitionedFile {
   public:
    int num_partitions_;       /**< Number of partitions in the file */
    int64_t partition_size_;   /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;       /**< Number of elements in each embedding */
    int64_t total_embeddings_; /**< Total number of embeddings */
    torch::Dtype dtype_;       /**< Datatype of the embeddings */
    int dtype_size_;           /**< Size in bytes of embedding element dtype */
    string filename_;          /**< Name of the backing file */
    int fd_;                   /**< File descriptor for the backing file */

    /** Constructor */
    PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings, torch::Dtype dtype);

    /** Loads a partition of the specified id into addr, assumes that addr has been allocated with sufficient memory */
    void readPartition(void *addr, Partition *partition);

    /** Writes a partition from memory to the file */
    void writePartition(Partition *partition, bool clear_mem = true);
};

class LookaheadBlock {
   private:
    std::thread *thread_;
    PartitionedFile *partitioned_file_;
    std::vector<void *> mems_;

    void run();

   public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    std::vector<Partition *> partitions_;
    std::atomic<bool> done_;

    LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_lookahead);

    ~LookaheadBlock();

    void start(std::vector<Partition *> first_partitions);

    void stop();

    void move_to_buffer(std::vector<void *> buff_addrs, std::vector<int64_t> buffer_idxs, std::vector<Partition *> next_partitions);
};

class AsyncWriteBlock {
   private:
    std::thread *thread_;
    PartitionedFile *partitioned_file_;
    std::vector<void *> mems_;

    void run();

   public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    std::vector<Partition *> partitions_;
    std::atomic<bool> done_;

    AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_evict);

    ~AsyncWriteBlock();

    void start();

    void stop();

    void async_write(std::vector<Partition *> partitions);
};

class PartitionBuffer {
   private:
    std::atomic<int64_t> size_;
    int capacity_;
    int num_partitions_;     /**< Number of partitions in the file */
    int64_t partition_size_; /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;     /**< Number of elements in each embedding */
    int fine_to_coarse_ratio_;
    int64_t total_embeddings_;
    torch::Dtype dtype_; /**< Datatype of the embeddings */
    int dtype_size_;

    void *buff_mem_;
    bool loaded_;

    Indices in_buffer_ids_;
    torch::Tensor buffer_tensor_view_;
    std::vector<Partition *> partition_table_;

    bool prefetching_;
    LookaheadBlock *lookahead_block_;
    AsyncWriteBlock *async_write_block_;

    string filename_;
    PartitionedFile *partitioned_file_;

    // order in which data is accessed
    torch::Tensor buffer_state_;
    std::vector<torch::Tensor> buffer_states_;
    std::vector<torch::Tensor>::iterator buffer_state_iterator_;

    torch::Tensor getBufferState();

    void admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs);

    void evict(std::vector<Partition *> evict_partitions);

    void startThreads();

    void stopThreads();

   public:
    PartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                    torch::Dtype dtype, string filename, bool prefetching);

    ~PartitionBuffer();

    void load();

    void write();

    void unload(bool write);

    std::vector<int> getNextAdmit();

    std::vector<int> getNextEvict();

    Indices getRandomIds(int64_t size);

    torch::Tensor indexRead(torch::Tensor indices);

    torch::Tensor getGlobalToLocalMap(bool get_current);

    void indexAdd(torch::Tensor indices, torch::Tensor values);

    void setBufferOrdering(std::vector<torch::Tensor> buffer_states);

    bool hasSwap();

    void performNextSwap();

    void sync();

    int64_t getNumInMemory() { return buffer_tensor_view_.size(0); }
};

#endif  // MARIUS_BUFFER_H
