//
// Created by Jason Mohoney on 4/21/20.
//

#include <storage.h>

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

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    exit(-1);
}

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
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = torch::typeMetaToScalarType(data.dtype());
    loaded_ = false;
    append(data);
    initialized_ = true;

}

FlatFile::FlatFile(string filename, torch::ScalarType dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
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
