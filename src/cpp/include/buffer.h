//
// Created by Jason Mohoney on 5/26/20.
//

#ifndef MARIUS_BUFFER_H
#define MARIUS_BUFFER_H

#include "batch.h"
#include "datatypes.h"

class Partition {
public:
    std::mutex *lock_;                                              /**< Mutex lock to prevent race conditions */
    std::condition_variable *cv_;                                   /**< Condition variable for signaling */
    void *data_ptr_;                                                /**< Pointer to partition in memory */
    int partition_id_;                                              /**< ID of the partition */

    bool present_;                                                  /**< If true this partition is present in the buffer */
    int usage_;                                                     /**< Counter which indicates the number of batches which are currently accessing this partition */

    int64_t partition_size_;                                        /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;                                            /**< Number of elements in each embedding */
    torch::Dtype dtype_;                                            /**< Datatype of the embeddings */
    int dtype_size_;                                                /**< Size in bytes of the datatype */
    int64_t total_size_;                                            /**< Total size in bytes of the partition */

    int64_t idx_offset_;                                            /**< Embedding ID offset of the partition */
    int64_t file_offset_;                                           /**< Offset in bytes of the partition in the embedding file */
    int buffer_idx_;                                                /**< Buffer entry ID of the partition in the buffer */

    torch::Tensor tensor_;                                          /**< Tensor view of the partition */

    bool evicting_;

    Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset);

    ~Partition();

    bool checkPresentAndIncrementUsage();

    void indexAddAndDecrementUsage(Indices indices, torch::Tensor value);

    torch::Tensor indexRead(Indices indices);

};

class PartitionedFile {
  public:
    int num_partitions_;                                            /**< Number of partitions in the file */
    int64_t partition_size_;                                        /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;                                            /**< Number of elements in each embedding */
    int64_t total_embeddings_;                                      /**< Total number of embeddings */
    torch::Dtype dtype_;                                            /**< Datatype of the embeddings */
    int dtype_size_;                                                /**< Size in bytes of embedding element dtype */
    string filename_;                                               /**< Name of the backing file */
    int fd_;                                                        /**< File descriptor for the backing file */

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
    void *mem_;

    void run();

  public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    Partition *partition_;
    std::atomic<bool> done_;

    LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file);

    ~LookaheadBlock();

    void start(Partition *first_partition);

    void stop();

    void move_to_buffer(void *addr, int64_t buffer_idx, Partition *next_partition);
};

class AsyncWriteBlock {
  private:
    std::thread *thread_;
    PartitionedFile *partitioned_file_;
    void *mem_;

    void run();

  public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    Partition *partition_;
    std::atomic<bool> done_;

    AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file);

    ~AsyncWriteBlock();

    void start();

    void stop();

    void async_write(Partition *partition);
};

class PartitionBuffer {
  private:
    std::atomic<int64_t> size_;
    int capacity_;
    int num_partitions_;                                            /**< Number of partitions in the file */
    int64_t partition_size_;                                        /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;                                            /**< Number of elements in each embedding */
    int64_t total_embeddings_;
    torch::Dtype dtype_;                                            /**< Datatype of the embeddings */
    int dtype_size_;

    // counters
    std::atomic<int> hits_;
    std::atomic<int> misses_;
    std::atomic<int> prefetch_hits_;
    
    void *buff_mem_;
    bool loaded_;
    torch::Tensor buffer_tensor_view_;
    std::vector<Partition *> partition_table_;
    std::queue<int> free_list_;

    bool prefetching_;
    LookaheadBlock *lookahead_block_;
    AsyncWriteBlock *async_write_block_;

    string filename_;
    PartitionedFile *partitioned_file_;

    // order in which data is accessed
    std::vector<int> ordering_;
    std::vector<int> evict_ids_;
    std::vector<int> admit_ids_;
    std::vector<int> admit_access_ids_;

    // order in which data is accessed
    std::vector<int>::iterator ordering_itr_;
    std::vector<int>::iterator evict_ids_itr_;
    std::vector<int>::iterator admit_ids_itr_;
    std::vector<int>::iterator admit_access_ids_itr_;

    // concurrency control
    std::mutex buffer_lock_;
    std::mutex access_lock_;
    std::condition_variable access_cv_;
    std::mutex admit_lock_;
    std::condition_variable admit_cv_;
    std::atomic<int> accesses_before_admit_;

    std::vector<int> getBufferState();

    // Filters out the indices which fall in partitions that have been evicted from the buffer
    torch::Tensor filterEvictedNegatives(std::vector<int> previous_buffer_state, torch::Tensor indices);

    // Will admit the partition if it not currently present in the buffer
    void admitIfNotPresent(int64_t access_id, Partition *partition);

    // Wait until this access is ready to be performed
    void waitRead(int64_t access_id);

    // Wait until this access is ready to be performed
    void waitAdmit(int64_t access_id);

    void admit(Partition *partition);

    void evict(Partition *partition);


  public:

    PartitionBuffer(int capacity, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings, torch::Dtype dtype, string filename);

    ~PartitionBuffer();

    void load();

    void unload(bool write);

    torch::Tensor indexRead(int partition_id, torch::Tensor indices, int64_t access_id);

    void indexAdd(int partition_id, torch::Tensor indices, torch::Tensor values);

    void bufferIndexAdd(std::vector<int> buffer_state, torch::Tensor indices, torch::Tensor values);

    std::tuple<std::vector<int>, torch::Tensor> bufferIndexRead(torch::Tensor indices);

    int64_t getHits() {
        return hits_;
    }

    int64_t getMisses() {
        return misses_;
    }

    int64_t getPrefetchHits() {
        return prefetch_hits_;
    }

    int64_t getSize() {
        return size_;
    }

    int64_t getTotalEmbeddings() {
        return total_embeddings_;
    };

    int64_t getBufferEmbeddingsCapacity() {
        return capacity_ * partition_size_;
    }

    // Get Evictions
    std::vector<Batch *> shuffleBeforeEvictions(std::vector<Batch *> batches);

    void setOrdering(std::vector<Batch *> batches);

    void sync();

    void startThreads();

    void stopThreads();

};


#endif //MARIUS_BUFFER_H
