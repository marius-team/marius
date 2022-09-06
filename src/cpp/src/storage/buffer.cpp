//
// Created by Jason Mohoney on 6/3/20.
//

#include "storage/buffer.h"

#include <common/util.h>
#include <fcntl.h>
#include <unistd.h>

#include <functional>
#include <future>
#include <shared_mutex>

#include "configuration/constants.h"
#include "reporting/logger.h"

Partition::Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset) {
    lock_ = new std::mutex();
    cv_ = new std::condition_variable();
    data_ptr_ = nullptr;
    partition_id_ = partition_id;

    present_ = false;

    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
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

torch::Tensor Partition::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    lock_->lock();

    torch::Tensor ret = tensor_.index_select(0, indices - idx_offset_);

    lock_->unlock();
    cv_->notify_all();

    return ret;
}

PartitionedFile::PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                                 torch::Dtype dtype) {
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);

    filename_ = filename;

    int flags = O_RDWR | IO_FLAGS;
    fd_ = open(filename_.c_str(), flags);
    if (fd_ == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void PartitionedFile::readPartition(void *addr, Partition *partition) {
    if (addr == NULL || partition == NULL) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    memset_wrapper(addr, 0, partition->total_size_);
    if (pread_wrapper(fd_, addr, partition->total_size_, partition->file_offset_) == -1) {
        SPDLOG_ERROR("Unable to read Block: {}\nError: {}", partition->partition_id_, errno);
        throw std::runtime_error("");
    }
    partition->data_ptr_ = addr;
    partition->tensor_ = torch::from_blob(addr, {partition->partition_size_, embedding_size_}, dtype_);
}

// writePartition accesses data pointed to by p->data_ptr_. Address p->data_ptr_ is expected to contain
// same data as that of p->tensor_.
void PartitionedFile::writePartition(Partition *partition, bool clear_mem) {
    if (partition == NULL || partition->data_ptr_ == nullptr) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    if (pwrite_wrapper(fd_, partition->data_ptr_, partition->total_size_, partition->file_offset_) == -1) {
        throw MariusRuntimeException(fmt::format("Unable to write partition: {}\nError: {}", partition->partition_id_, errno));
    }

    if (clear_mem) {
        memset_wrapper(partition->data_ptr_, 0, partition->total_size_);
        partition->data_ptr_ = nullptr;
        partition->tensor_ = torch::Tensor();
    }
}

LookaheadBlock::LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_lookahead) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;
    partitions_ = {};
    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_lookahead);

    for (int i = 0; i < num_per_lookahead; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

LookaheadBlock::~LookaheadBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void LookaheadBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == false; });

        if (partitions_.empty()) {
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            std::unique_lock partition_lock(*partition->lock_);
            partition->cv_->wait(partition_lock, [partition] { return partition->evicting_ == false; });
            partitioned_file_->readPartition(mems_[i], partition);
            partition_lock.unlock();
            partition->cv_->notify_all();
        }

        present_ = true;
        lock.unlock();
        cv_.notify_all();
    }
}

void LookaheadBlock::start(std::vector<Partition *> first_partitions) {
    partitions_ = first_partitions;
    if (thread_ == nullptr) {
        thread_ = new std::thread(&LookaheadBlock::run, this);
    }
}

void LookaheadBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = false;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void LookaheadBlock::move_to_buffer(std::vector<void *> buff_addrs, std::vector<int64_t> buffer_idxs, std::vector<Partition *> next_partitions) {
    if (partitions_.size() > buff_addrs.size() || partitions_.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }
    // wait until block is populated
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == true; });

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        Partition *partition = partitions_[i];
        void *addr = buff_addrs[i];
        int64_t buffer_idx = buffer_idxs[i];
        memcpy_wrapper(addr, mems_[i], partition->total_size_);
        memset_wrapper(mems_[i], 0, partition->total_size_);

        partition->data_ptr_ = addr;
        partition->tensor_ = torch::from_blob(partition->data_ptr_, {partition->partition_size_, partition->embedding_size_}, partition->dtype_);
        partition->buffer_idx_ = buffer_idx;
        partition->present_ = true;
    }

    // next partition will be prefetched automatically
    partitions_ = next_partitions;
    present_ = false;
    lock.unlock();
    cv_.notify_all();
}

AsyncWriteBlock::AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_evict) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;

    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_evict);

    for (int i = 0; i < num_per_evict; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

AsyncWriteBlock::~AsyncWriteBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void AsyncWriteBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == true; });

        if (done_) {
            return;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            partitioned_file_->writePartition(partition);
            partition->present_ = false;
            partition->evicting_ = false;
            partition->cv_->notify_all();
        }

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

void AsyncWriteBlock::async_write(std::vector<Partition *> partitions) {
    if (partitions.size() > mems_.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }

    // wait until block is empty
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == false; });

    partitions_ = partitions;

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        void *mem = mems_[i];
        Partition *partition = partitions_[i];

        memcpy_wrapper(mem, partition->data_ptr_, total_size_);
        memset_wrapper(partition->data_ptr_, 0, total_size_);

        partition->data_ptr_ = mem;
        partition->evicting_ = true;
    }

    present_ = true;

    lock.unlock();
    cv_.notify_all();
}

PartitionBuffer::PartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size,
                                 int64_t total_embeddings, torch::Dtype dtype, string filename, bool prefetching) {
    capacity_ = capacity;
    size_ = 0;
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    fine_to_coarse_ratio_ = fine_to_coarse_ratio;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    filename_ = filename;
    partition_table_ = std::vector<Partition *>();

    prefetching_ = prefetching;

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

    filename_ = filename;
    partitioned_file_ = new PartitionedFile(filename_, num_partitions_, partition_size_, embedding_size_, total_embeddings_, dtype_);

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
            throw std::runtime_error("");
        }
        memset_wrapper(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);
        buffer_tensor_view_ = torch::from_blob(buff_mem_, {capacity_ * partition_size_, embedding_size_}, dtype_);

        // initialize buffer
        int partition_id;

        int64_t num_nodes = 0;

        for (int i = 0; i < buffer_state_.size(0); i++) {
            partition_id = buffer_state_[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            void *buff_addr = (char *)buff_mem_ + (i * partition_size_ * embedding_size_ * dtype_size_);
            partitioned_file_->readPartition(buff_addr, partition);
            partition->present_ = true;
            partition->buffer_idx_ = i;
            num_nodes += partition->partition_size_;
        }

        in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);
        //        int64_t offset = 0;
        //        for (int i = 0; i < buffer_state_.size(0); i++) {
        //            partition_id = buffer_state_[i].item<int>();
        //            Partition *partition = partition_table_[partition_id];
        //            int64_t partition_offset = partition->idx_offset_;
        //
        //            in_buffer_ids_.slice(0, offset, offset + partition->partition_size_) = torch::arange(partition_offset, partition_offset +
        //            partition->partition_size_); offset += partition->partition_size_;
        //        }

        if (prefetching_) {
            lookahead_block_ = new LookaheadBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
            async_write_block_ = new AsyncWriteBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
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

torch::Tensor PartitionBuffer::getBufferState() { return buffer_state_; }

// indices a relative to the local node ids
torch::Tensor PartitionBuffer::indexRead(torch::Tensor indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    return buffer_tensor_view_.index_select(0, indices);
}

Indices PartitionBuffer::getRandomIds(int64_t size) { return torch::randint(in_buffer_ids_.size(0), size, torch::kInt64); }

// indices must contain unique values, else there is a possibility of a race condition
void PartitionBuffer::indexAdd(torch::Tensor indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || buffer_tensor_view_.size(1) != values.size(1)) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    // buffer_tensor_view_.index_add_(0, indices, values);

    // assumes this operation is only used on float valued data, and this op takes place on the CPU
    auto data_accessor = buffer_tensor_view_.accessor<float, 2>();
    auto ids_accessor = indices.accessor<int64_t, 1>();
    auto values_accessor = values.accessor<float, 2>();

    int d = values.size(1);
    int64_t size = indices.size(0);
#pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        for (int j = 0; j < d; j++) {
            data_accessor[ids_accessor[i]][j] += values_accessor[i][j];
        }
    }
}

void PartitionBuffer::setBufferOrdering(std::vector<torch::Tensor> buffer_states) {
    buffer_states_ = buffer_states;
    buffer_state_iterator_ = buffer_states_.begin();
    buffer_state_ = *buffer_state_iterator_++;

    if (loaded_) {
        unload(true);
        load();
    }
}

bool PartitionBuffer::hasSwap() { return buffer_state_iterator_ != buffer_states_.end(); }

void PartitionBuffer::performNextSwap() {
    if (!buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end()) {
        return;
    }

    // get evicted and admitted partitions
    std::vector<int> evict_ids = getNextEvict();
    std::vector<int> admit_ids = getNextAdmit();

    std::vector<Partition *> admit_partitions;
    std::vector<Partition *> evict_partitions;
    std::vector<int64_t> evict_buffer_idxs;
    for (int admit_id : admit_ids) {
        admit_partitions.emplace_back(partition_table_[admit_id]);
    }
    for (int evict_id : evict_ids) {
        evict_partitions.emplace_back(partition_table_[evict_id]);
        evict_buffer_idxs.emplace_back(partition_table_[evict_id]->buffer_idx_);
    }

    buffer_state_ = *buffer_state_iterator_++;

    // evict partition
    evict(evict_partitions);
    // admit partition
    admit(admit_partitions, evict_buffer_idxs);

    int64_t num_nodes = 0;

    int partition_id;
    for (int i = 0; i < buffer_state_.size(0); i++) {
        partition_id = buffer_state_[i].item<int>();
        num_nodes += partition_table_[partition_id]->partition_size_;
    }

    in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);

    //    int64_t offset = 0;
    //    for (int i = 0; i < buffer_state_.size(0); i++) {
    //        partition_id = buffer_state_[i].item<int>();
    //        Partition *partition = partition_table_[partition_id];
    //        int64_t partition_offset = partition->idx_offset_;
    //
    //        in_buffer_ids_.slice(0, offset, offset + partition->partition_size_) = torch::arange(partition_offset, partition_offset +
    //        partition->partition_size_); offset += partition->partition_size_;
    //    }
}

std::vector<int> PartitionBuffer::getNextAdmit() {
    std::vector<int> admit_ids;
    bool admitted;

    if (buffer_state_iterator_ != buffer_states_.end()) {
        for (int i = 0; i < buffer_state_iterator_->size(0); i++) {
            admitted = true;
            for (int j = 0; j < buffer_state_.size(0); j++) {
                if ((*buffer_state_iterator_)[i].item<int>() == (buffer_state_)[j].item<int>()) {
                    admitted = false;
                }
            }
            if (admitted) {
                admit_ids.emplace_back((*buffer_state_iterator_)[i].item<int>());
            }
        }
    }
    return admit_ids;
}

std::vector<int> PartitionBuffer::getNextEvict() {
    std::vector<int> evict_ids;
    bool evicted;

    for (int i = 0; i < buffer_state_.size(0); i++) {
        evicted = true;
        for (int j = 0; j < buffer_state_iterator_->size(0); j++) {
            if ((*buffer_state_iterator_)[j].item<int>() == buffer_state_[i].item<int>()) {
                evicted = false;
            }
        }
        if (evicted) {
            evict_ids.emplace_back(buffer_state_[i].item<int>());
        }
    }
    return evict_ids;
}

torch::Tensor PartitionBuffer::getGlobalToLocalMap(bool get_current) {
    torch::Tensor buffer_index_map = -torch::ones({total_embeddings_}, torch::kInt64);

    torch::Tensor buffer_state;

    if (get_current) {
        buffer_state = buffer_state_;

#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;
            int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
        }

    } else {
        // get mapping for next swap
        buffer_state = *buffer_state_iterator_;

        // get evicted and admitted partitions
        std::vector<int> evict_ids = getNextEvict();
        std::vector<int> admit_ids = getNextAdmit();

        // get mapping for the partitions that will still be in the buffer
#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;

            if (partition->buffer_idx_ != -1) {
                int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
                buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                    torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
            }
        }

// get mapping for the partitions that will be admitted
#pragma omp parallel for
        for (int i = 0; i < evict_ids.size(); i++) {
            Partition *admit_partition = partition_table_[admit_ids[i]];
            Partition *evict_partition = partition_table_[evict_ids[i]];
            int64_t partition_offset = admit_partition->idx_offset_;
            int64_t buffer_offset = evict_partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + admit_partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + admit_partition->partition_size_);
        }
    }
    return buffer_index_map;
}

void PartitionBuffer::evict(std::vector<Partition *> evict_partitions) {
    if (prefetching_) {
        async_write_block_->async_write(evict_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < evict_partitions.size(); i++) {
            partitioned_file_->writePartition(evict_partitions[i]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < evict_partitions.size(); i++) {
        evict_partitions[i]->present_ = false;
    }
}

void PartitionBuffer::admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs) {
    if (admit_partitions.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    std::vector<void *> buff_addrs(buffer_idxs.size());

#pragma omp parallel for
    for (int i = 0; i < buffer_idxs.size(); i++) {
        void *buff_addr = (char *)buff_mem_ + (buffer_idxs[i] * partition_size_ * embedding_size_ * dtype_size_);
        buff_addrs[i] = buff_addr;
    }

    if (prefetching_) {
        std::vector<int> next_admit_ids = getNextAdmit();
        std::vector<Partition *> next_partitions;
        if (!next_admit_ids.empty()) {
            for (int admit_id : next_admit_ids) {
                next_partitions.emplace_back(partition_table_[admit_id]);
            }
        }
        lookahead_block_->move_to_buffer(buff_addrs, buffer_idxs, next_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < admit_partitions.size(); i++) {
            Partition *partition = admit_partitions[i];
            partitioned_file_->readPartition(buff_addrs[i], partition);
            partition->present_ = true;
            partition->buffer_idx_ = buffer_idxs[i];
        }
    }
}

void PartitionBuffer::sync() {
    SPDLOG_DEBUG("Synchronizing buffer");
    Partition *curr_partition;
    for (int i = 0; i < num_partitions_; i++) {
        curr_partition = partition_table_[i];
        if (curr_partition->present_) {
            partitioned_file_->writePartition(curr_partition, true);
            curr_partition->present_ = false;
            curr_partition->buffer_idx_ = -1;
        }
    }
}

void PartitionBuffer::startThreads() {
    SPDLOG_DEBUG("Starting prefetching threads");
    std::vector<Partition *> partitions;
    std::vector<int> admit_ids = getNextAdmit();
    for (int admit_id : admit_ids) {
        partitions.emplace_back(partition_table_[admit_id]);
    }
    lookahead_block_->start(partitions);
    async_write_block_->start();
}

void PartitionBuffer::stopThreads() {
    SPDLOG_DEBUG("Stopping prefetching threads");
    lookahead_block_->stop();
    async_write_block_->stop();
}
