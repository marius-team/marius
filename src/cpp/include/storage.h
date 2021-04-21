//
// Created by Jason Mohoney on 4/21/20.
//


#ifndef MARIUS_STORAGE_H
#define MARIUS_STORAGE_H
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <batch.h>
#include <buffer.h>
#include <config.h>
#include <datatypes.h>
#include <decoder.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <util.h>
#include <vector>

using std::vector;
using std::string;
using std::shared_ptr;
using std::list;
using std::unordered_map;

/** Abstract storage class */
class Storage {
  protected:
    int64_t dim0_size_;
    int64_t dim1_size_;
    torch::ScalarType dtype_;
    bool initialized_;
    vector<int64_t> edge_bucket_sizes_;

  public:
    virtual ~Storage() {};

    virtual torch::Tensor indexRead(Indices indices) = 0;

    virtual void indexAdd(Indices indices, torch::Tensor values) = 0;

    virtual torch::Tensor range(int64_t offset, int64_t n) = 0;

    virtual void indexPut(Indices indices, torch::Tensor values) = 0;

    virtual void load() = 0;

    virtual void unload(bool write = false) = 0;

    virtual void checkpoint(int epoch_id) = 0;

    virtual void shuffle() = 0;

    int64_t getDim0() {
        return dim0_size_;
    }

    bool isInitialized() {
        return initialized_;
    }

    void setInitialized(bool init) {
        initialized_ = init;
    }

    void readPartitionSizes(string filename) {
        std::ifstream partition_file(filename);
        edge_bucket_sizes_.clear();
        int64_t size;
        while (partition_file >> size) {
            edge_bucket_sizes_.push_back(size);
        }
    }

    vector<int64_t> getEdgeBucketSizes() {
        return edge_bucket_sizes_;
    }
};

/** Storage which uses the partition buffer, used for node embeddings and optimizer state */
class PartitionBufferStorage : public Storage {
  protected:
    string filename_;
    bool loaded_;

    int64_t partition_size_;

    int64_t num_partitions_;

    PartitionBuffer *buffer_;

    int64_t capacity_;

    bool is_embeddings_;

  public:
    PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype, int64_t capacity, bool embeddings);

    PartitionBufferStorage(string filename, torch::Tensor data, int64_t capacity, bool embeddings);

    PartitionBufferStorage(string filename, int64_t capacity, bool embeddings);

    ~PartitionBufferStorage();

    void rangePut(int64_t offset, torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void shuffle() override {
        SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
        exit(-1);
    };

    torch::Tensor indexRead(int partition_id, Indices indices, int64_t access_id);

    void indexAdd(int partition_id, Indices indices, torch::Tensor values);

    torch::Tensor range(int partition_id, int64_t offset, int64_t n);

    void bufferIndexAdd(std::vector<int> buffer_state, torch::Tensor indices, torch::Tensor values);

    std::tuple<std::vector<int>, torch::Tensor> bufferIndexRead(torch::Tensor indices);

    void sync() {
        buffer_->sync();
    }

    vector<Batch *> shuffleBeforeEvictions(vector<Batch *> batches) {
        return buffer_->shuffleBeforeEvictions(batches);
    }

    void setOrdering(vector<Batch *> batches) {
        buffer_->setOrdering(batches);
    }

    int64_t getHits() {
        return buffer_->getHits();
    }

    int64_t getMisses() {
        return buffer_->getMisses();
    }

    int64_t getPrefetchHits() {
        return buffer_->getPrefetchHits();
    }

    int64_t getBufferSize() {
        return buffer_->getSize();
    }

    int64_t getPartitionSize() {
        return partition_size_;
    }

    int64_t getBufferEmbeddingsCapacity() {
        return buffer_->getBufferEmbeddingsCapacity();
    }


};

/** Flat File storage used for data that only requires sequential access. Can be used to store and access large amounts of edges. */
class FlatFile : public Storage {
  private:
    string filename_;

    int fd_;

    torch::Tensor data_;

    bool loaded_;

  public:
    FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype);

    FlatFile(string filename, torch::Tensor data);

    FlatFile(string filename, torch::ScalarType dtype);

    ~FlatFile() {};

    void rangePut(int64_t offset, torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void shuffle() override;

    void move(string new_filename);

    void copy(string new_filename, bool rename);

    void mem_load();

    void mem_unload(bool write);
};

/** Memory mapped storage for large amounts of data that require both sequential and random access. */
class MemoryMap : public Storage {
private:
    string filename_;

    int fd_;

    torch::Tensor data_;

    bool loaded_;

    int block_size_;

public:
    MemoryMap(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype);

    MemoryMap(string filename, torch::Tensor data);

    MemoryMap(string filename, torch::ScalarType dtype);

    void rangePut(int64_t offset, torch::Tensor values);

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void mem_load();

    void mem_unload(bool write);

    void shuffle() override;
};

/** In memory storage for data which fits in either GPU or CPU memory. */
class InMemory : public Storage {
  private:
    string filename_;

    int fd_;

    torch::Tensor data_;

    bool loaded_;

    torch::DeviceType device_;

  public:
    InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype, torch::DeviceType device);

    InMemory(string filename, torch::Tensor data, torch::DeviceType device);

    InMemory(string filename, torch::ScalarType dtype);

    ~InMemory() {};

    void load() override;

    void unload(bool write) override;

    void checkpoint(int epoch_id) override;

    torch::Tensor indexRead(Indices indices) override;

    void indexAdd(Indices indices, torch::Tensor values) override;

    torch::Tensor range(int64_t offset, int64_t n) override;

    void indexPut(Indices indices, torch::Tensor values) override;

    void force_load();

    void shuffle() override;
};

#endif //MARIUS_STORAGE_H
