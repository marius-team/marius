//
// Created by Jason Mohoney on 6/3/20.
//

#include "buffer.h"

#include <fcntl.h>
#include <unistd.h>
#include <util.h>

#include <functional>
#include <future>
#include <shared_mutex>

#include "config.h"
#include "logger.h"


Partition::Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset) {

    lock_ = new std::mutex();
    cv_ = new std::condition_variable();
    data_ptr_ = nullptr;
    partition_id_ = partition_id;

    present_ = false;
    usage_ = 0;

    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    dtype_ = dtype;
    if (dtype_ == torch::kFloat64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size_ = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size_ = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size_ = 4;
    }
    total_size_ = partition_size_ * embedding_size_ * dtype_size_;

    idx_offset_ = idx_offset;
    file_offset_ = file_offset;
    buffer_idx_ = -1;

    tensor_ = torch::Tensor();

    evicting_ = false;
}

Partition::~Partition() {
    delete lock_;
    delete cv_;
    tensor_ = torch::Tensor();
}

bool Partition::checkPresentAndIncrementUsage() {
    bool present;
    lock_->lock();

    present = present_;
    usage_++;

    lock_->unlock();
    cv_->notify_all();
    return present;
}

void Partition::indexAddAndDecrementUsage(Indices indices, torch::Tensor value) {
    lock_->lock();

    tensor_.index_add_(0, indices - idx_offset_, value);
    usage_--;

    lock_->unlock();
    cv_->notify_all();
}

torch::Tensor Partition::indexRead(Indices indices) {
    lock_->lock();

    torch::Tensor ret = tensor_.index_select(0, indices - idx_offset_);

    lock_->unlock();
    cv_->notify_all();

    return ret;
}

PartitionedFile::PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings, torch::Dtype dtype) {
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    dtype_ = dtype;
    if (dtype_ == torch::kFloat64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size_ = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size_ = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size_ = 4;
    }

    filename_ = filename;

    int flags = O_RDWR | IO_FLAGS;
    fd_ = open(filename_.c_str(), flags);
    if (fd_ == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        exit(-1);
    }
}

void PartitionedFile::readPartition(void* addr, Partition *partition) {
    memset(addr, 0, partition->total_size_);
    if (pread(fd_, addr, partition->total_size_, partition->file_offset_) == -1) {
        SPDLOG_ERROR("Unable to read Block: {}\nError: {}", partition->partition_id_, errno);
        exit(-1);
    }
    partition->data_ptr_ = addr;
    partition->tensor_ = torch::from_blob(addr, {partition->partition_size_, embedding_size_}, dtype_);
}

void PartitionedFile::writePartition(Partition *partition, bool clear_mem) {
    if (pwrite(fd_, partition->data_ptr_, partition->total_size_, partition->file_offset_) == -1) {
        SPDLOG_ERROR("Unable to write Block: {}\nError: {}", partition->partition_id_, errno);
        exit(-1);
    }

    if (clear_mem) {
        memset(partition->data_ptr_, 0, partition->total_size_);
        partition->data_ptr_ = nullptr;
        partition->tensor_ = torch::Tensor();
    }
}

LookaheadBlock::LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;
    partition_ = nullptr;
    lock_ = new std::mutex();

    if(posix_memalign(&mem_, 4096, total_size_)) {
        SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
        exit(-1);
    }
    memset(mem_, 0, total_size_);

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

LookaheadBlock::~LookaheadBlock() {
    delete lock_;
    free(mem_);
}

void LookaheadBlock::run() {
    while(!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == false; });

        if (partition_ == nullptr) {
            break;
        }

        std::unique_lock partition_lock(*partition_->lock_);
        partition_->cv_->wait(partition_lock, [this] {return partition_->evicting_ == false;});
        partitioned_file_->readPartition(mem_, partition_);
        partition_lock.unlock();
        partition_->cv_->notify_all();

        present_ = true;
        lock.unlock();
        cv_.notify_all();
    }
}

void LookaheadBlock::start(Partition *first_partition) {
    partition_ = first_partition;
    if (thread_ == nullptr) {
        thread_ = new std::thread(&LookaheadBlock::run, this);
    }
}

void LookaheadBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            thread_->join();
        }
        delete thread_;
    }
}

void LookaheadBlock::move_to_buffer(void *addr, int64_t buffer_idx, Partition *next_partition) {

    // wait until block is populated
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == true; });

    memcpy(addr, mem_, partition_->total_size_);
    memset(mem_, 0, partition_->total_size_);

    partition_->data_ptr_ = addr;
    partition_->tensor_ = torch::from_blob(partition_->data_ptr_, {partition_->partition_size_, partition_->embedding_size_}, partition_->dtype_);
    partition_->buffer_idx_ = buffer_idx;
    partition_->present_ = true;

    // next partition will be prefetched automatically
    partition_ = next_partition;
    present_ = false;
    lock.unlock();
    cv_.notify_all();
}

AsyncWriteBlock::AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;

    lock_ = new std::mutex();

    if(posix_memalign(&mem_, 4096, total_size_)) {
        SPDLOG_ERROR("Unable to allocate async evict memory\nError: {}", errno);
        exit(-1);
    }
    memset(mem_, 0, total_size_);

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

AsyncWriteBlock::~AsyncWriteBlock() {
    delete lock_;
    free(mem_);
}


void AsyncWriteBlock::run() {
    while(!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == true; });

        if (done_) {
            return;
        }

        std::unique_lock partition_lock(*partition_->lock_);
        partition_->cv_->wait(partition_lock, [this] { return partition_->usage_ == 0; });
        partitioned_file_->writePartition(partition_);
        partition_->present_ = false;
        partition_->evicting_ = false;
        partition_lock.unlock();
        partition_->cv_->notify_all();

        present_ = false;
        lock.unlock();
        cv_.notify_all();
    }
}

void AsyncWriteBlock::start() {
    if (thread_ == nullptr) {
        thread_ = new std::thread(&AsyncWriteBlock::run, this);
    }
}

void AsyncWriteBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = true;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void AsyncWriteBlock::async_write(Partition *partition) {

    // wait until block is empty
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == false; });

    partition_ = partition;
    memcpy(mem_, partition_->data_ptr_, total_size_);
    memset(partition_->data_ptr_, 0, total_size_);

    partition_->data_ptr_ = mem_;
    partition->evicting_ = true;
    present_ = true;

    lock.unlock();
    cv_.notify_all();
}


PartitionBuffer::PartitionBuffer(int capacity, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings, torch::Dtype dtype, string filename) {
    capacity_ = capacity;
    size_ = 0;
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    dtype_ = dtype;
    if (dtype_ == torch::kFloat64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size_ = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size_ = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size_ = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size_ = 4;
    }

    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    filename_ = filename;

    hits_ = 0;
    misses_ = 0;
    prefetch_hits_ = 0;

    free_list_ = std::queue<int>();
    partition_table_ = std::vector<Partition *>();

    prefetching_ = marius_options.storage.prefetching;

    int64_t curr_idx_offset = 0;
    int64_t curr_file_offset = 0;
    int64_t curr_partition_size = partition_size_;
    int64_t curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
    for (int64_t i = 0; i < num_partitions_; i++) {

        // the last partition might be slightly smaller
        if (i == num_partitions_ - 1) {
            curr_partition_size = total_embeddings_ - curr_idx_offset;
            curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
        }

        Partition *curr_part = new Partition(i, curr_partition_size, embedding_size_, dtype_, curr_idx_offset, curr_file_offset);
        partition_table_.push_back(curr_part);

        curr_file_offset += curr_total_size;
        curr_idx_offset += curr_partition_size;
    }

    for (int64_t i = 0; i < capacity_; i++) {
        free_list_.push(i);
    }

    filename_ = filename;
    partitioned_file_ = new PartitionedFile(filename_, num_partitions_, partition_size_, embedding_size_, total_embeddings_, dtype_);

    accesses_before_admit_ = 0;
    loaded_ = false;
}

PartitionBuffer::~PartitionBuffer() {

    unload(true);

    delete partitioned_file_;
    for (int64_t i = 0; i < num_partitions_; i++) {
        delete partition_table_[i];
    }
}

void PartitionBuffer::load() {
    if (!loaded_) {

        if (posix_memalign(&buff_mem_, 4096, capacity_ * partition_size_ * embedding_size_ * dtype_size_)) {
            SPDLOG_ERROR("Unable to allocate buffer memory\nError: {}", errno);
            exit(-1);
        }
        memset(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);
        buffer_tensor_view_ = torch::from_blob(buff_mem_, {capacity_ * partition_size_, embedding_size_}, dtype_);

        if (prefetching_) {
            lookahead_block_ = new LookaheadBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_);
            async_write_block_ = new AsyncWriteBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_);
            startThreads();
        }

        loaded_ = true;
    }
}

void PartitionBuffer::unload(bool write) {
    if (loaded_) {

        if (write) {
            sync();
        }
        buffer_tensor_view_ = torch::Tensor();
        free(buff_mem_);
        buff_mem_ = nullptr;

        if (prefetching_) {
            stopThreads();
            delete lookahead_block_;
            delete async_write_block_;
        }

        size_ = 0;
        loaded_ = false;
    }
}

std::vector<int> PartitionBuffer::getBufferState() {
    std::vector<int> buffer_state(capacity_);
    for (int i = 0; i < capacity_; i++) {
        buffer_state[i] = -1;
    }
    for (int i = 0; i < partition_table_.size(); i++) {
        if (partition_table_[i]->present_) {
            buffer_state[partition_table_[i]->buffer_idx_] = partition_table_[i]->partition_id_;
        }
    }
    return buffer_state;
}

torch::Tensor PartitionBuffer::filterEvictedNegatives(std::vector<int> previous_buffer_state, torch::Tensor indices) {
    std::vector<int> current_buffer_state = getBufferState();
    torch::Tensor filter_mask = torch::ones_like(indices);
    for (int i = 0; i < previous_buffer_state.size(); i++) {
        if (current_buffer_state[i] != previous_buffer_state[i]) {
            int64_t ignore_start = i * partition_size_;
            int64_t ignore_end = ignore_start + partition_size_;

            std::vector<int64_t> indices_vec(indices.data_ptr<int64_t>(), indices.data_ptr<int64_t>() + indices.size(0));
            int64_t ignore_idx_start = std::lower_bound(indices_vec.begin(), indices_vec.end(), ignore_start) - indices_vec.begin();
            int64_t length = std::lower_bound(indices_vec.begin(), indices_vec.end(), ignore_end) - indices_vec.begin() - ignore_idx_start;
            filter_mask.narrow(0, ignore_idx_start, length) = 0;
        }
    }
    return filter_mask.to(torch::kBool);
}

void PartitionBuffer::waitRead(int64_t access_id) {
    std::unique_lock access_lock(access_lock_);
    access_cv_.wait(access_lock, [this, access_id] { return access_id <= *admit_access_ids_itr_; });
    access_lock.unlock();
    access_cv_.notify_all();
}

void PartitionBuffer::waitAdmit(int64_t access_id) {
    std::unique_lock access_lock(access_lock_);
    access_cv_.wait(access_lock, [this, access_id] { return (access_id == *admit_access_ids_itr_) && (accesses_before_admit_ == 0); });
    access_lock.unlock();
    access_cv_.notify_all();
}

void PartitionBuffer::admitIfNotPresent(int64_t access_id, Partition *partition) {
    waitRead(access_id);
    assert(loaded_);
    if (!partition->checkPresentAndIncrementUsage()) {
        waitAdmit(access_id);
        if (!partition->present_) {
            admit_lock_.lock();
            admit(partition);
            admit_lock_.unlock();
        }
    }
    access_lock_.lock();
    accesses_before_admit_--;
    if (access_id % 2 == 0 && ordering_[access_id] == ordering_[access_id + 1]) {
        accesses_before_admit_--;
    }
    access_lock_.unlock();
    access_cv_.notify_all();
}

// indices a relative to the global embedding structure
torch::Tensor PartitionBuffer::indexRead(int partition_id, torch::Tensor indices, int64_t access_id) {
    Partition *partition = partition_table_[partition_id];
    admitIfNotPresent(access_id, partition);
    // locking the partition/buffer not necessary since partition is guaranteed to be in the buffer
    return partition->indexRead(indices);
}

void PartitionBuffer::indexAdd(int partition_id, torch::Tensor indices, torch::Tensor values) {
    // locking the partition/buffer not necessary since partition is guaranteed to be in the buffer
    Partition *partition = partition_table_[partition_id];
    partition->indexAddAndDecrementUsage(indices, values);
}

void PartitionBuffer::bufferIndexAdd(std::vector<int> buffer_state, torch::Tensor indices, torch::Tensor values) {
    // buffer needs to be locked to prevent evictions during an update
    buffer_lock_.lock();
    torch::Tensor filter = filterEvictedNegatives(buffer_state, indices);
    indices = indices.masked_select(filter);
    if (indices.size(0) == 0) {
        buffer_lock_.unlock();
        return;
    }
    values = values.masked_select(filter.unsqueeze(-1)).reshape({indices.size(0), -1});
    buffer_tensor_view_.index_add_(0, indices, values);
    buffer_lock_.unlock();
}

std::tuple<std::vector<int>, torch::Tensor> PartitionBuffer::bufferIndexRead(torch::Tensor indices) {
    // buffer needs to be locked to prevent evictions during a read
    buffer_lock_.lock();
    Embeddings ret = buffer_tensor_view_.index_select(0, indices);
    std::vector<int> buffer_state = getBufferState();
    buffer_lock_.unlock();
    return std::forward_as_tuple(buffer_state, ret);
}

void PartitionBuffer::evict(Partition *partition) {
    if (marius_options.storage.prefetching) {
        std::unique_lock partition_lock(*partition->lock_);
        partition->cv_->wait(partition_lock, [partition] { return partition->usage_ == 0; });
        async_write_block_->async_write(partition);
        partition->present_ = false;
        partition_lock.unlock();
        partition->cv_->notify_all();
    } else {
        std::unique_lock partition_lock(*partition->lock_);
        partition->cv_->wait(partition_lock, [partition] { return partition->usage_ == 0; });
        partitioned_file_->writePartition(partition);
        partition->present_ = false;
        partition_lock.unlock();
        partition->cv_->notify_all();
    }
    assert(partition->buffer_idx_ >= 0);
    assert(partition->buffer_idx_ < capacity_);
    free_list_.push(partition->buffer_idx_);
}

void PartitionBuffer::admit(Partition *partition) {
    // assumes that the buffer has been locked but not the partition
    int64_t buffer_idx;
    SPDLOG_TRACE("Admitting {}", partition->partition_id_);
    if (free_list_.empty()) {
        Partition *partition_to_evict = partition_table_[*evict_ids_itr_++];
        SPDLOG_TRACE("Evicting {}", partition_to_evict->partition_id_);
        evict(partition_to_evict);
        size_--;
        SPDLOG_TRACE("Evicted {}", partition_to_evict->partition_id_);
    }
    buffer_idx = free_list_.front();
    free_list_.pop();

    void *buff_addr = (char *) buff_mem_ + (buffer_idx * partition_size_ * embedding_size_ * dtype_size_);

    if (marius_options.storage.prefetching) {
        Partition *next_partition = nullptr;
        if (admit_ids_itr_ != admit_ids_.end()) {
            next_partition = partition_table_[*admit_ids_itr_++];
        }
        lookahead_block_->move_to_buffer(buff_addr, buffer_idx, next_partition);
    } else {
        partition->lock_->lock();
        partitioned_file_->readPartition(buff_addr, partition);
        partition->present_ = true;
        partition->buffer_idx_ = buffer_idx;
        partition->lock_->unlock();
    }

    size_++;
    access_lock_.lock();
    accesses_before_admit_ = *(admit_access_ids_itr_ + 1) - *admit_access_ids_itr_;
    admit_access_ids_itr_++;
    access_lock_.unlock();
    access_cv_.notify_all();
    SPDLOG_TRACE("Admitted {}", partition->partition_id_);
}

void PartitionBuffer::sync() {
    SPDLOG_DEBUG("Synchronizing buffer");
    Partition *curr_partition;
    for (int i = 0; i < num_partitions_; i++) {
        curr_partition = partition_table_[i];
        if (curr_partition->present_) {
            partitioned_file_->writePartition(curr_partition, true);
            curr_partition->present_ = false;
            free_list_.push(curr_partition->buffer_idx_);
            curr_partition->buffer_idx_ = -1;
        }
    }
}

std::vector<Batch *> PartitionBuffer::shuffleBeforeEvictions(std::vector<Batch *> batches) {

    std::vector<int64_t> ordering;

    std::map<int64_t, int64_t> buffer_sim;
    std::vector<int64_t> evictions;

    int64_t i = 0;
    for (auto itr = batches.begin(); itr != batches.end(); itr++) {
        int64_t s = ((PartitionBatch *) (*itr))->src_partition_idx_;
        int64_t d = ((PartitionBatch *) (*itr))->dst_partition_idx_;

        ordering.emplace_back(s);
        ordering.emplace_back(d);
    }

    i = 0;
    for (auto itr = ordering.begin(); itr != ordering.end(); itr++) {
        if (buffer_sim.find(*itr) != buffer_sim.end()) {
            // hit
        } else {
            // miss
            if ((int) buffer_sim.size() == capacity_) {
                int64_t max_distance = 0;
                int64_t distance = 0;
                int64_t evict_id = 0;
                for (int64_t k = 0; k < num_partitions_; k++) {
                    if (buffer_sim.find(k) != buffer_sim.end()) {
                        if (i % 2 == 1 && k == ordering[i - 1]) {
                            continue;
                        }
                        // find distance
                        distance = 0;
                        for (int64_t j = i % ordering.size(); j < (int64_t) ordering.size(); j++) {
                            if (ordering[j] == k) {
                                break;
                            }
                            distance++;
                        }
                        if (distance >= max_distance) {
                            max_distance = distance;
                            evict_id = k;
                        }
                    }
                }
                evictions.emplace_back(i);

                buffer_sim.erase(evict_id);
            }
            buffer_sim.insert(std::pair(*itr, 0));
        }
        i++;
    }

    int64_t prev_idx = 0;
    int64_t evict_batch_id = 0;
    for (auto itr = evictions.begin(); itr != evictions.end(); itr++) {
        evict_batch_id = *itr / 2;

        if (evict_batch_id - prev_idx > 2) {
            std::vector<Batch *> tmp_batches(evict_batch_id - prev_idx - 1);

            torch::Tensor rand_idx = torch::randperm(evict_batch_id - prev_idx - 1, torch::kInt64);
            for (int j = 0; j < (evict_batch_id - prev_idx - 1); j++) {
                tmp_batches[rand_idx[j].item<int64_t>()] = batches[prev_idx + j];
            }

            for (int j = 0; j < (evict_batch_id - prev_idx - 1); j++) {
                batches[prev_idx + j] = tmp_batches[j];
            }
        }

        prev_idx = evict_batch_id + 1;
    }

    int64_t k = 0;
    for (auto itr = batches.begin(); itr != batches.end(); itr++) {
        batches[k]->batch_id_ = k;
        k++;
    }

    return batches;
}

void PartitionBuffer::setOrdering(std::vector<Batch *> batches) {
    ordering_.clear();
    evict_ids_.clear();
    admit_ids_.clear();
    admit_access_ids_.clear();

    for (auto itr = batches.begin(); itr != batches.end(); itr++) {
        int64_t s = ((PartitionBatch *) (*itr))->src_partition_idx_;
        int64_t d = ((PartitionBatch *) (*itr))->dst_partition_idx_;

        ordering_.emplace_back(s);
        ordering_.emplace_back(d);
    }

    std::map<int64_t, int64_t> buffer_sim;
    std::vector<int64_t> evictions;
    int64_t i = 0;
    bool admitted = false;
    while ((int) buffer_sim.size() < capacity_) {
        admitted = buffer_sim.insert(std::pair(ordering_[i++], 0)).second;
        if (admitted) {
            admit_ids_.emplace_back(ordering_[i - 1]);
            admit_access_ids_.emplace_back(i - 1);
        }
    }

    i = 0;
    for (auto itr = ordering_.begin(); itr != ordering_.end(); itr++) {
        bool swap = false;
        int64_t evict_id = -1;
        if (buffer_sim.find(*itr) != buffer_sim.end()) {
            // hit
        } else {
            // miss
            if ((int) buffer_sim.size() == capacity_) {
                int64_t max_distance = 0;
                int64_t distance = 0;
                for (int64_t k = 0; k < num_partitions_; k++) {
                    if (buffer_sim.find(k) != buffer_sim.end()) {
                        if (i % 2 == 1 && k == ordering_[i - 1]) {
                            continue;
                        }
                        // find distance
                        distance = 0;
                        for (int64_t j = i % ordering_.size(); j < (int64_t) ordering_.size(); j++) {
                            if (ordering_[j] == k) {
                                break;
                            }
                            distance++;
                        }
                        if (distance >= max_distance) {
                            max_distance = distance;
                            evict_id = k;
                        }
                    }
                }
                evictions.emplace_back(i);

                buffer_sim.erase(evict_id);
                swap = true;
            }
            buffer_sim.insert(std::pair(*itr, 0));
        }
        if (swap) {
            evict_ids_.emplace_back(evict_id);
            admit_ids_.emplace_back(ordering_[i]);
            admit_access_ids_.emplace_back(i);
        }
        i++;
    }
    admit_access_ids_.emplace_back(i);

    evict_ids_itr_ = evict_ids_.begin();
    admit_ids_itr_ = admit_ids_.begin();
    admit_access_ids_itr_ = admit_access_ids_.begin();
    accesses_before_admit_ = *admit_access_ids_itr_;
}

void PartitionBuffer::startThreads() {
    lookahead_block_->start(partition_table_[*admit_ids_itr_++]);
    async_write_block_->start();
}

void PartitionBuffer::stopThreads() {
    lookahead_block_->stop();
    async_write_block_->stop();
}
