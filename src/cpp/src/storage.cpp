//
// Created by Jason Mohoney on 4/21/20.
//

#include "storage.h"

#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>

#ifdef MARIUS_OMP
#include <omp.h>
#endif

#include "config.h"
#include "logger.h"
#include "util.h"

using std::ios;
using std::ios_base;

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype, int64_t capacity, bool embeddings) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    capacity_ = capacity;
    num_partitions_ = marius_options.storage.num_partitions;
    partition_size_ = ceil((double) dim0_size_ / num_partitions_);
    is_embeddings_ = embeddings;

    buffer_ = new PartitionBuffer(capacity_, num_partitions_, partition_size_, dim1_size_, dim0_size_, dtype_, filename_);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, int64_t capacity, bool embeddings) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = torch::typeMetaToScalarType(data.dtype());
    append(data);
    initialized_ = true;
    loaded_ = false;
    capacity_ = capacity;
    num_partitions_ = marius_options.storage.num_partitions;
    partition_size_ = ceil((double) dim0_size_ / num_partitions_);
    is_embeddings_ = embeddings;

    buffer_ = new PartitionBuffer(capacity_, num_partitions_, partition_size_, dim1_size_, dim0_size_, dtype_, filename_);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t capacity, bool embeddings) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = marius_options.storage.embeddings_dtype;
    capacity_ = capacity;
    num_partitions_ = marius_options.storage.num_partitions;
    partition_size_ = ceil((double) dim0_size_ / num_partitions_);
    is_embeddings_ = embeddings;

    buffer_ = new PartitionBuffer(capacity_, num_partitions_, partition_size_, dim1_size_, dim0_size_, dtype_, filename_);
}

void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        exit(-1);
    }

    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        exit(-1);
    }

    close(fd);
}

void PartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = torch::typeMetaToScalarType(values.dtype());
    num_partitions_ = marius_options.storage.num_partitions;

    std::ofstream outfile(filename_, flags);

    int dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    outfile.write((char *) values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();

    partition_size_ = ceil((double) dim0_size_ / num_partitions_);
}

PartitionBufferStorage::~PartitionBufferStorage() {
    delete buffer_;
}

void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}

void PartitionBufferStorage::unload(bool write) {
    if (loaded_) {
        buffer_->unload(write);
        loaded_ = false;
    }
}

void PartitionBufferStorage::checkpoint(int epoch_id) {
    if (loaded_) {
        buffer_->sync();
    }
    string output_path = filename_.substr(0, filename_.size() - PathConstants::file_ext.size()) + "_" + std::to_string(epoch_id) + PathConstants::file_ext;
    std::ifstream src;
    std::ofstream dst;

    src.open(filename_, ios::in | ios::binary);
    dst.open(output_path, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

torch::Tensor PartitionBufferStorage::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

std::tuple<torch::Tensor, torch::Tensor> PartitionBufferStorage::gatherNeighbors(torch::Tensor node_ids, bool src) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

void PartitionBufferStorage::initializeInMemorySubGraph(std::vector<int> buffer_state) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

void PartitionBufferStorage::updateInMemorySubGraph(int admit_partition_id, int evict_partition_id) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    exit(-1);
};


torch::Tensor PartitionBufferStorage::indexRead(int partition_id, Indices indices, int64_t access_id) {
    return buffer_->indexRead(partition_id, indices, access_id);
}

void PartitionBufferStorage::indexAdd(int partition_id, Indices indices, torch::Tensor values) {
    buffer_->indexAdd(partition_id, indices, values);
}

torch::Tensor PartitionBufferStorage::range(int partition_id, int64_t offset, int64_t n) {
    return buffer_->indexRead(partition_id, torch::arange(offset, n, torch::kInt64), 0);
}

void PartitionBufferStorage::bufferIndexAdd(std::vector<int> buffer_state, torch::Tensor indices, torch::Tensor values) {
    buffer_->bufferIndexAdd(buffer_state, indices, values);
}

std::tuple<std::vector<int>, torch::Tensor> PartitionBufferStorage::bufferIndexRead(torch::Tensor indices) {
    return buffer_->bufferIndexRead(indices);
}

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;

    in_memory_subgraph_enabled_ = false;
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = torch::typeMetaToScalarType(data.dtype());
    loaded_ = false;
    append(data);
    initialized_ = true;

    in_memory_subgraph_enabled_ = false;
}

FlatFile::FlatFile(string filename, torch::ScalarType dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;

    in_memory_subgraph_enabled_ = false;
}

void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        exit(-1);
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = torch::typeMetaToScalarType(values.dtype());

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    outfile.write((char *) values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}


void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            exit(-1);
        }
        loaded_ = true;
    }
}

void FlatFile::unload(bool write) {
    (void) write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

void FlatFile::checkpoint(int epoch_id) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, use copy instead");
    exit(-1);
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    exit(-1);
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    exit(-1);
}

void FlatFile::initializeInMemorySubGraph(std::vector<int> buffer_state) {

    in_memory_subgraph_enabled_ = true;
    std::vector<torch::Tensor> in_memory_edge_buckets;
    std::vector<int> in_memory_edge_bucket_ids;
    std::vector<int> in_memory_edge_bucket_starts;
    std::vector<int> in_memory_edge_bucket_sizes;

    torch::Tensor new_in_memory_partition_ids = torch::from_blob(buffer_state.data(), {(int) buffer_state.size()}, torch::kInt32);

    torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int) edge_bucket_sizes_.size()}, torch::kInt64);
    edge_bucket_sizes = edge_bucket_sizes.to(torch::kInt32);
    torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
    torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;

    int curr_in_memory_start = 0;
    // new partition as src
    for (int i = 0; i < new_in_memory_partition_ids.size(0); i++) {
        {
            int edge_bucket_id = (new_in_memory_partition_ids[i].item<int>() * marius_options.storage.num_partitions) + new_in_memory_partition_ids[i].item<int>();
            int edge_bucket_start = edge_bucket_starts_disk[edge_bucket_id].item<int>();
            int edge_bucket_size = edge_bucket_sizes[edge_bucket_id].item<int>();

            torch::Tensor edge_bucket = range(edge_bucket_start, edge_bucket_size);
            in_memory_edge_buckets.emplace_back(edge_bucket);
            in_memory_edge_bucket_ids.emplace_back(edge_bucket_id);
            in_memory_edge_bucket_starts.emplace_back(curr_in_memory_start);
            in_memory_edge_bucket_sizes.emplace_back(edge_bucket_size);
            curr_in_memory_start += edge_bucket_size;
        }
        for (int j = 0; j < new_in_memory_partition_ids.size(0); j++) {
            if (i != j) {
                int edge_bucket_id = (new_in_memory_partition_ids[i].item<int>() * marius_options.storage.num_partitions) + new_in_memory_partition_ids[j].item<int>();
                int edge_bucket_start = edge_bucket_starts_disk[edge_bucket_id].item<int>();
                int edge_bucket_size = edge_bucket_sizes[edge_bucket_id].item<int>();

                torch::Tensor edge_bucket = range(edge_bucket_start, edge_bucket_size);
                in_memory_edge_buckets.emplace_back(edge_bucket);
                in_memory_edge_bucket_ids.emplace_back(edge_bucket_id);
                in_memory_edge_bucket_starts.emplace_back(curr_in_memory_start);
                in_memory_edge_bucket_sizes.emplace_back(edge_bucket_size);
                curr_in_memory_start += edge_bucket_size;
            }
        }
    }
    in_memory_subgraph_ = torch::cat(in_memory_edge_buckets);

    // sort by source
    src_sorted_list_ = in_memory_subgraph_.index_select(0, torch::argsort(in_memory_subgraph_.select(1, 0)));

    // sort by dest
    dst_sorted_list_ = in_memory_subgraph_.index_select(0, torch::argsort(in_memory_subgraph_.select(1, 2)));

    // update state
    in_memory_partition_ids_ = new_in_memory_partition_ids.clone();
    in_memory_edge_bucket_ids_ = torch::from_blob(in_memory_edge_bucket_ids.data(), {(int) in_memory_edge_bucket_ids.size()}, torch::kInt32).clone();
    in_memory_edge_bucket_starts_ = torch::from_blob(in_memory_edge_bucket_starts.data(), {(int) in_memory_edge_bucket_starts.size()}, torch::kInt32).clone();
    in_memory_edge_bucket_sizes_ = torch::from_blob(in_memory_edge_bucket_sizes.data(), {(int) in_memory_edge_bucket_sizes.size()}, torch::kInt32).clone();
}


std::tuple<torch::Tensor, torch::Tensor> FlatFile::gatherNeighbors(torch::Tensor node_ids, bool src) {

    std::vector<torch::Tensor> neighbors = std::vector<torch::Tensor>(node_ids.size(0));

    torch::Tensor search_values = torch::ones_like(node_ids);
    search_values = node_ids + search_values;
    search_values = torch::stack({node_ids, search_values}).transpose(0, 1);

    torch::Tensor ranges;

    if (src) {
        ranges = torch::searchsorted(src_sorted_list_.select(1, 0), search_values);
    } else {
        ranges = torch::searchsorted(dst_sorted_list_.select(1, 2), search_values);
    }

    torch::Tensor num_neighbors = ranges.select(1, 1) - ranges.select(1, 0);
    torch::Tensor offsets = num_neighbors.cumsum(0) - num_neighbors;
    int total_neighbors = torch::sum(num_neighbors).item<int64_t>();

    int64_t *neighbor_id_mem = (int64_t *) malloc(total_neighbors * sizeof(int64_t));
    auto starts = ranges.select(1, 0);
    auto ends = ranges.select(1, 1);

    auto starts_accessor = starts.accessor<int64_t, 1>();
    auto ends_accessor = ends.accessor<int64_t, 1>();
    auto offsets_accessor = offsets.accessor<int64_t, 1>();
    auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();


    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < node_ids.size(0); i++) {
            int64_t offset = offsets_accessor[i];

            int count = 0;
            for (int64_t j = starts_accessor[i]; j < ends_accessor[i]; j++) {
                *(neighbor_id_mem + offset + count) = j;
                count++;
            }
        }
    }

    torch::Tensor neighbor_ids = torch::from_blob(neighbor_id_mem,  {total_neighbors}, torch::kInt64);

    std::tuple<torch::Tensor, torch::Tensor> ret;

    if (src) {
        ret = std::forward_as_tuple(src_sorted_list_.index_select(0, neighbor_ids), offsets);
    } else {
        ret = std::forward_as_tuple(dst_sorted_list_.index_select(0, neighbor_ids), offsets);
    }

    return ret;
}

void FlatFile::updateInMemorySubGraph(int admit_partition_id, int evict_partition_id) {
    if (in_memory_subgraph_enabled_) {
        std::vector<torch::Tensor> in_memory_edge_buckets;
        std::vector<int> in_memory_edge_bucket_ids;
        std::vector<int> in_memory_edge_bucket_starts;
        std::vector<int> in_memory_edge_bucket_sizes;

        // get edge buckets that will be kept in memory
        torch::Tensor src_partitions = torch::floor(in_memory_edge_bucket_ids_ / marius_options.storage.num_partitions);
        torch::Tensor dst_partitions = in_memory_edge_bucket_ids_ % marius_options.storage.num_partitions;
        torch::Tensor keep_mask = (src_partitions != evict_partition_id) & (dst_partitions != evict_partition_id);

        torch::Tensor kept_edge_buckets_ids = in_memory_edge_bucket_ids_.masked_select(keep_mask);
        torch::Tensor kept_edge_buckets_starts = in_memory_edge_bucket_starts_.masked_select(keep_mask);
        torch::Tensor kept_edge_buckets_sizes = in_memory_edge_bucket_sizes_.masked_select(keep_mask);

        // copy existing edge buckets
        in_memory_edge_buckets.reserve(kept_edge_buckets_starts.size(0));
        for (int i = 0; i < kept_edge_buckets_starts.size(0); i++) {
            in_memory_edge_buckets.emplace_back(in_memory_subgraph_.narrow(0, kept_edge_buckets_starts[i].item<int>(), kept_edge_buckets_sizes[i].item<int>()));
            in_memory_edge_bucket_ids.emplace_back(kept_edge_buckets_ids.item<int>());
            in_memory_edge_bucket_starts.emplace_back(kept_edge_buckets_starts[i].item<int>());
            in_memory_edge_bucket_sizes.emplace_back(kept_edge_buckets_sizes[i].item<int>());
        }

        torch::Tensor new_in_memory_partition_ids = torch::cat({in_memory_partition_ids_.masked_select(in_memory_partition_ids_ != admit_partition_id), torch::full({1}, admit_partition_id)});

        torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int) edge_bucket_sizes_.size()});
        torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
        torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;


        int curr_in_memory_start = kept_edge_buckets_starts[-1].item<int>();
        // new partition as src
        for (int i = 0; i < new_in_memory_partition_ids.size(0) - 1; i++) {
            int edge_bucket_id = (admit_partition_id * marius_options.storage.num_partitions) + new_in_memory_partition_ids[i].item<int>();
            int edge_bucket_start = edge_bucket_starts_disk[edge_bucket_id].item<int>();
            int edge_bucket_size = edge_bucket_sizes[edge_bucket_id].item<int>();

            torch::Tensor edge_bucket = range(edge_bucket_start, edge_bucket_size);
            in_memory_edge_buckets.emplace_back(edge_bucket);
            in_memory_edge_bucket_ids.emplace_back(edge_bucket_id);
            in_memory_edge_bucket_starts.emplace_back(curr_in_memory_start);
            in_memory_edge_bucket_sizes.emplace_back(edge_bucket_size);
            curr_in_memory_start += edge_bucket_size;
        }

        // new partition as dst
        for (int i = 0; i < new_in_memory_partition_ids.size(0) - 1; i++) {
            int edge_bucket_id = (new_in_memory_partition_ids[i].item<int>() * marius_options.storage.num_partitions) + admit_partition_id;
            int edge_bucket_start = edge_bucket_starts_disk[edge_bucket_id].item<int>();
            int edge_bucket_size = edge_bucket_sizes[edge_bucket_id].item<int>();

            torch::Tensor edge_bucket = range(edge_bucket_start, edge_bucket_size);
            in_memory_edge_buckets.emplace_back(edge_bucket);
            in_memory_edge_bucket_ids.emplace_back(edge_bucket_id);
            in_memory_edge_bucket_starts.emplace_back(curr_in_memory_start);
            in_memory_edge_bucket_sizes.emplace_back(edge_bucket_size);
            curr_in_memory_start += edge_bucket_size;
        }

        // add self edge bucket
        {
            int edge_bucket_id = (admit_partition_id * marius_options.storage.num_partitions) + admit_partition_id;
            int edge_bucket_start = edge_bucket_starts_disk[edge_bucket_id].item<int>();
            int edge_bucket_size = edge_bucket_sizes[edge_bucket_id].item<int>();

            torch::Tensor edge_bucket = range(edge_bucket_start, edge_bucket_size);
            in_memory_edge_buckets.emplace_back(edge_bucket);
            in_memory_edge_bucket_ids.emplace_back(edge_bucket_id);
            in_memory_edge_bucket_starts.emplace_back(curr_in_memory_start);
            in_memory_edge_bucket_sizes.emplace_back(edge_bucket_size);
            curr_in_memory_start += edge_bucket_size;
        }

        in_memory_subgraph_ = torch::cat(in_memory_edge_buckets);

        // sort by source
        src_sorted_list_ = in_memory_subgraph_.index_select(0, torch::argsort(in_memory_subgraph_.select(1, 0)));

        // sort by dest
        dst_sorted_list_ = in_memory_subgraph_.index_select(0, torch::argsort(in_memory_subgraph_.select(1, 2)));

        // update state
        in_memory_partition_ids_ = new_in_memory_partition_ids.clone();
        in_memory_edge_bucket_ids_ = torch::from_blob(in_memory_edge_bucket_ids.data(), {(int) in_memory_edge_bucket_ids.size()}).clone();
        in_memory_edge_bucket_starts_ = torch::from_blob(in_memory_edge_bucket_starts.data(), {(int) in_memory_edge_bucket_starts.size()}).clone();
        in_memory_edge_bucket_sizes_ = torch::from_blob(in_memory_edge_bucket_sizes.data(), {(int) in_memory_edge_bucket_sizes.size()}).clone();
    }
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    exit(-1);
}

void FlatFile::move(string new_filename) {
    unload(false);
    std::filesystem::rename(filename_, new_filename);
    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);
    std::filesystem::copy_file(filename_, new_filename, std::filesystem::copy_options::update_existing);
    if (rename) {
        filename_ = new_filename;
    }
    load();
}

torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    int dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);

    if (pread(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        exit(-1);
    }

    return output_tensor;
}

void FlatFile::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while(offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SHUFFLE_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SHUFFLE_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::randperm(chunk.size(0), opts)));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr + 1 != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}


void FlatFile::mem_load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            exit(-1);
        }

        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        int64_t curr_bytes = 0;

        while (offset < read_size) {
            curr_bytes = read_size - offset;
            if (curr_bytes > 1e9) {
                curr_bytes = 1e9;
            }

            if (pread(fd_, (char *) data_.data_ptr() + offset, curr_bytes, offset) == -1) {
                SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
                exit(-1);
            }

            offset += curr_bytes;
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = 0;
        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        int64_t curr_bytes = 0;

        if (write) {
            while (offset < read_size) {
                curr_bytes = read_size - offset;
                if (curr_bytes > 1e9) {
                    curr_bytes = 1e9;
                }

                if (pwrite(fd_, (char *) data_.data_ptr() + offset, curr_bytes, offset) == -1) {
                    SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                    exit(-1);
                }
                offset += curr_bytes;
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Edges written");
        process_mem_usage();
        loaded_ = false;
        process_mem_usage();
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Nulled tensor and pointer");
        process_mem_usage();
    }
}

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::ScalarType dtype, torch::DeviceType device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::DeviceType device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = torch::typeMetaToScalarType(data.dtype());
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = 0;

    if (dtype_ == torch::kFloat64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kFloat32) {
        dtype_size = 4;
    } else if (dtype_ == torch::kFloat16) {
        dtype_size = 2;
    } else if (dtype_ == torch::kInt64) {
        dtype_size = 8;
    } else if (dtype_ == torch::kInt32) {
        dtype_size = 4;
    }

    outfile.write((char *) temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::ScalarType dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

void InMemory::initializeInMemorySubGraph(std::vector<int> buffer_state) {
    // sort by source
    src_sorted_list_ = data_.index_select(0, torch::argsort(data_.select(1, 0)));

    // sort by dest
    dst_sorted_list_ = data_.index_select(0, torch::argsort(data_.select(1, 2)));
}

std::tuple<torch::Tensor, torch::Tensor> InMemory::gatherNeighbors(torch::Tensor node_ids, bool src) {
    std::vector<torch::Tensor> neighbors = std::vector<torch::Tensor>(node_ids.size(0));

    torch::Tensor search_values = torch::ones_like(node_ids);
    search_values = node_ids + search_values;
    search_values = torch::stack({node_ids, search_values}).transpose(0, 1);

    torch::Tensor ranges;

    if (src) {
        ranges = torch::searchsorted(src_sorted_list_.select(1, 0), search_values);
    } else {
        ranges = torch::searchsorted(dst_sorted_list_.select(1, 2), search_values);
    }

    torch::Tensor num_neighbors = ranges.select(1, 1) - ranges.select(1, 0);
    torch::Tensor offsets = num_neighbors.cumsum(0) - num_neighbors;
    int total_neighbors = torch::sum(num_neighbors).item<int>();
    torch::Tensor neighbor_ids = torch::empty({total_neighbors});

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < node_ids.size(0); i++) {
            neighbor_ids.narrow(0, offsets[i], num_neighbors[i].item<int>()).copy_(torch::arange(ranges.select(0, 0).item<int>(), ranges.select(0, 1).item<int>()));
        }
    }


    return std::forward_as_tuple(src_sorted_list_.gather(0, neighbor_ids), offsets);
}

void InMemory::updateInMemorySubGraph(int admit_partition_id, int evict_partition_id) {}

void InMemory::load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            exit(-1);
        }

        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        int64_t curr_bytes = 0;

        while (offset < read_size) {
            curr_bytes = read_size - offset;
            if (curr_bytes > 1e9) {
                curr_bytes = 1e9;
            }

            if (pread(fd_, (char *) data_.data_ptr() + offset, curr_bytes, offset) == -1) {
                SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
                exit(-1);
            }

            offset += curr_bytes;
        }

        if (device_ == torch::kCUDA) {
            data_ = data_.to(device_);
        }

        loaded_ = true;
    }
}

void InMemory::force_load() {
    if (!loaded_) {
        load();
    } else {
        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        int64_t curr_bytes = 0;

        while (offset < read_size) {
            curr_bytes = read_size - offset;
            if (curr_bytes > 1e9) {
                curr_bytes = 1e9;
            }

            if (pread(fd_, (char *) data_.data_ptr() + offset, curr_bytes, offset) == -1) {
                SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
                exit(-1);
            }
            offset += curr_bytes;
        }
        data_ = data_.to(device_);

        loaded_ = true;
    }
}

void InMemory::unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = 0;

        if (device_ == torch::kCUDA) {
            data_ = data_.to(torch::kCPU);
        }

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;
        int64_t curr_bytes = 0;

        if (write) {
            while (offset < read_size) {
                curr_bytes = read_size - offset;
                if (curr_bytes > 1e9) {
                    curr_bytes = 1e9;
                }

                if (pwrite(fd_, (char *) data_.data_ptr() + offset, curr_bytes, offset) == -1) {
                    SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
                    exit(-1);
                }
                offset += curr_bytes;
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Written");
        process_mem_usage();

        loaded_ = false;
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Done");
        process_mem_usage();
    }
}

void InMemory::checkpoint(int epoch_id) {
    if (loaded_) {
        unload(true);
    }
    string output_path = filename_.substr(0, filename_.size() - PathConstants::file_ext.size()) + "_" + std::to_string(epoch_id) + PathConstants::file_ext;
    std::ifstream src;
    std::ofstream dst;

    src.open(filename_, ios::in | ios::binary);
    dst.open(output_path, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

torch::Tensor InMemory::indexRead(Indices indices) {
    return data_.index_select(0, indices.to(device_));
}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    data_.index_add_(0, indices.to(device_), values.to(device_));
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    data_[indices.to(device_)] = values.to(device_);
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    return data_.narrow(0, offset, n);
}

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    } else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr + 1 != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
