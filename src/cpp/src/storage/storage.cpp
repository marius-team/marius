//
// Created by Jason Mohoney on 4/21/20.
//

#include "storage/storage.h"

#include <fcntl.h>
#include <unistd.h>

#include <iostream>

#include "common/util.h"
#include "configuration/constants.h"
#include "reporting/logger.h"

using std::ios;
using std::ios_base;

void renameFile(string old_filename, string new_filename) {
    int result = rename(old_filename.c_str(), new_filename.c_str());
    if (result != 0) {
        SPDLOG_ERROR("Unable to rename {}\nError: {}", old_filename, errno);
        throw std::runtime_error("");
    }
}

void copyFile(string src_file, string dst_file) {
    std::ifstream src;
    std::ofstream dst;

    src.open(src_file, ios::in | ios::binary);
    dst.open(dst_file, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

bool fileExists(string filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void createDir(string path, bool exist_ok) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
            if (exist_ok) {
                SPDLOG_DEBUG("{} directory already exists", path);
            } else {
                SPDLOG_ERROR("{} directory already exists", path);
                throw std::runtime_error("");
            }
        } else {
            SPDLOG_ERROR("Failed to create {}\nError: {}", path, errno);
            throw std::runtime_error("");
        }
    }
}

Storage::Storage() : device_(torch::kCPU) {}

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    options_ = options;
    dtype_ = options_->dtype;
    append(data);
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    options_ = options;
    dtype_ = options_->dtype;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
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
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

PartitionBufferStorage::~PartitionBufferStorage() { delete buffer_; }

void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}

void PartitionBufferStorage::write() {
    if (loaded_) {
        buffer_->sync();
    }
}

void PartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        buffer_->unload(perform_write);
        loaded_ = false;
    }
}

torch::Tensor PartitionBufferStorage::indexRead(Indices indices) { return buffer_->indexRead(indices); }

void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { return buffer_->indexAdd(indices, values); }

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

void PartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, bool alloc) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = torch::kCPU;

    if (alloc) {
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

        std::ofstream ofs(filename_, std::ios::binary | std::ios::out);
        ofs.seekp(dim0_size_ * dim1_size_ * dtype_size - 1);
        ofs.write("", 1);
        ofs.close();
    }
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    loaded_ = false;
    append(data);
    initialized_ = true;
    device_ = torch::kCPU;
}

FlatFile::FlatFile(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
}

void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    if (!values.defined() || (dim0_size_ != 0 && (values.size(0) + offset > dim0_size_ || values.size(1) != dim1_size_))) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags = dim0_size_ == 0 ? ios::trunc | ios::binary : ios::binary | ios_base::app;

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}

void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        loaded_ = true;
    }
}

void FlatFile::write() { return; }

void FlatFile::unload(bool perform_write) {
    (void)perform_write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::move(string new_filename) {
    unload(false);

    renameFile(filename_, new_filename);

    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);

    copyFile(filename_, new_filename);

    if (rename) {
        filename_ = new_filename;
    }
    load();
}

torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);
    if (pread_wrapper(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
    return output_tensor;
}

void FlatFile::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
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
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
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

void FlatFile::sort(bool src) {
    // function for sorting flat file storing edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SORT_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SORT_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::argsort(chunk.select(1, sort_dim))));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
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
            throw std::runtime_error("");
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
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

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::Device device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::Device device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

InMemory::InMemory(torch::Tensor data) {
    if (data.sizes().size() == 2) {
        dim0_size_ = data.size(0);
        dim1_size_ = data.size(1);
    } else if (data.sizes().size() == 1) {
        dim0_size_ = data.size(0);
        dim1_size_ = 1;
    } else {
        throw MariusRuntimeException("Tensor must have 1 or two dimensions");
    }

    filename_ = "";
    data_ = data.reshape({dim0_size_, dim1_size_});

    initialized_ = true;
    dtype_ = data.dtype().toScalarType();
    device_ = data.device();
    loaded_ = true;
}

void InMemory::load() {
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        if (device_ == torch::kCUDA) {
            data_ = data_.to(device_);
        }

        loaded_ = true;
    }
}

void InMemory::write() {
    if (loaded_ && !filename_.empty()) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        if (device_ == torch::kCUDA) {
            data = data_.to(torch::kCPU);
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void InMemory::unload(bool perform_write) {
    if (loaded_ && !filename_.empty()) {
        if (perform_write) {
            write();
        }

        close(fd_);
        loaded_ = false;
        data_ = torch::Tensor();
    }
}

torch::Tensor InMemory::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    if (data_.defined()) {
        return data_.index_select(0, indices.to(device_));
    } else {
        return torch::Tensor();
    }
}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_.index_add_(0, indices, values);
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
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
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_[indices] = values;
    } else {
        // assumes this operation is only used on float valued data.
        auto data_accessor = data_.accessor<float, 2>();
        auto ids_accessor = indices.accessor<int64_t, 1>();
        auto values_accessor = values.accessor<float, 2>();

        int d = values.size(1);
        int64_t size = indices.size(0);
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int j = 0; j < d; j++) {
                data_accessor[ids_accessor[i]][j] = values_accessor[i][j];
            }
        }
    }
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    return data_.narrow(0, offset, n);
}

void InMemory::rangePut(int64_t offset, int64_t n, torch::Tensor values) { data_.narrow(0, offset, n).copy_(values); }

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full shuffle
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    }
    // shuffle within edge buckets
    else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void InMemory::sort(bool src) {
    // function for sorting in memory edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full sort
    if (edge_bucket_sizes_.empty()) {
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::argsort(data_.select(1, sort_dim))));
    }
    // sort within edge buckets
    else {
        int64_t start = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
