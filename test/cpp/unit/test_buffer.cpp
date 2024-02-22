#include <fcntl.h>
#include <unistd.h>

#include <string>

#include "gtest/gtest.h"
#include "storage/storage.h"
#include "testing_util.h"
#include "util.h"

#define tryNextSwapAndAssert(pb, admits_, evicts_) \
    ASSERT_EQ(pb->hasSwap(), true);                \
    admits = pb->getNextAdmit();                   \
    evicts = pb->getNextEvict();                   \
    ASSERT_EQ(admits, admits_);                    \
    ASSERT_EQ(evicts, evicts_);                    \
    pb->performNextSwap();

class PartitionBufferTest : public ::testing::Test {
   protected:
    string filename;
    int fd;
    torch::Tensor rand_tensor_float32;
    int64_t rows;
    int64_t cols;
    int capacity;
    int num_partitions;
    int fine_to_coarse_ratio;
    int64_t partition_size;
    int embedding_size;
    int64_t total_embeddings;
    torch::Dtype dtype;
    int dtype_size;
    vector<torch::Tensor> buffer_states;
    PartitionBuffer *pb;

    PartitionBufferTest() {
        total_embeddings = 45;
        rows = 100;
        cols = 100;
        capacity = 2;
        num_partitions = 5;
        partition_size = 10;
        embedding_size = rows * cols;
        dtype = torch::kFloat32;
        dtype_size = get_dtype_size_wrapper(dtype);
        fine_to_coarse_ratio = 2;
    }

    ~PartitionBufferTest() {}

    void SetUp() override {
        filename = testing::TempDir() + "embeddings_data.txt";
        fd = createTmpFile(filename);
        ASSERT_NE(fd, -1);

        int64_t tensor_size = rows * cols * dtype_size;
        ASSERT_EQ(genRandTensorAndWriteToFile(rand_tensor_float32, total_embeddings, embedding_size, dtype, fd), total_embeddings * tensor_size);

        for (int i = 1; i < 5; i++) {
            torch::Tensor state = torch::zeros({2}, torch::kInt64);
            state[1] = i;
            buffer_states.push_back(state);
        }
        for (int i = 4; i >= 2; i--) {
            torch::Tensor state = torch::ones({2}, torch::kInt64);
            state[1] = i;
            buffer_states.push_back(state);
        }
        {
            torch::Tensor state = torch::zeros({2}, torch::kInt64);
            state[0] = 2;
            state[1] = 3;
            buffer_states.push_back(state);
        }
        {
            torch::Tensor state = torch::zeros({2}, torch::kInt64);
            state[0] = 2;
            state[1] = 4;
            buffer_states.push_back(state);
        }
        {
            torch::Tensor state = torch::zeros({2}, torch::kInt64);
            state[0] = 3;
            state[1] = 4;
            buffer_states.push_back(state);
        }
    }

    void initializePartitionBuffer(bool prefetch) {
        pb = new PartitionBuffer(capacity, num_partitions, fine_to_coarse_ratio, partition_size, embedding_size, total_embeddings, dtype, filename, prefetch);
        pb->setBufferOrdering(buffer_states);
        pb->load();
    }

    void TearDown() override {
        close(fd);
        remove(filename.c_str());
        delete (pb);
    }
};

class PartitionedFileTest : public ::testing::Test {
   protected:
    int num_partitions;
    int64_t partition_size;
    int embedding_size;
    int64_t total_embeddings;
    torch::Dtype dtype;
    int dtype_size;
    string filename;
    int fd;
    PartitionedFile *pf;
    torch::Tensor rand_tensor_float32;

    PartitionedFileTest() {
        num_partitions = 5;
        partition_size = 10;
        embedding_size = 10;
        total_embeddings = 45;
        dtype = torch::kFloat32;
        dtype_size = get_dtype_size_wrapper(dtype);
    }

    ~PartitionedFileTest() {}

    void SetUp() override {
        filename = testing::TempDir() + "partitioned_file.txt";
        fd = createTmpFile(filename);
        ASSERT_NE(fd, -1);
        ASSERT_EQ(genRandTensorAndWriteToFile(rand_tensor_float32, total_embeddings, embedding_size, dtype, fd),
                  total_embeddings * embedding_size * dtype_size);
        pf = new PartitionedFile(filename, num_partitions, partition_size, embedding_size, total_embeddings, dtype);
    }

    void TearDown() override {
        remove(filename.c_str());
        delete (pf);
        close(fd);
    }
};

class LookaheadBlockTest : public ::testing::Test {
   protected:
    int total_size;
    PartitionedFile *pf;
    int total_embeddings;
    int num_per_lookahead;
    int num_partitions;
    int partition_size;
    int embedding_size;
    int dtype_size;
    torch::Dtype dtype;
    string filename;
    int fd;
    torch::Tensor rand_tensor_float32;
    std::vector<Partition *> partitions;
    std::vector<Partition *> returned_partitions;

    LookaheadBlockTest() {
        total_embeddings = 45;
        num_partitions = 5;
        partition_size = 10;
        embedding_size = 10000;
        dtype = torch::kFloat32;
        dtype_size = get_dtype_size_wrapper(dtype);
        total_size = partition_size * embedding_size * dtype_size;
        num_per_lookahead = 2;
    }

    ~LookaheadBlockTest() {}

    void SetUp() override {
        filename = testing::TempDir() + "lookahead_buffer.txt";
        fd = createTmpFile(filename);
        ASSERT_NE(fd, -1);
        ASSERT_EQ(genRandTensorAndWriteToFile(rand_tensor_float32, total_embeddings, embedding_size, dtype, fd),
                  total_embeddings * embedding_size * dtype_size);
        pf = new PartitionedFile(filename, num_partitions, partition_size, embedding_size, total_embeddings, dtype);
    }

    void TearDown() override {
        remove(filename.c_str());
        delete (pf);
        close(fd);
        for (int i = 0; i < partitions.size(); i++) {
            delete (partitions[i]);
            delete (returned_partitions[i]);
        }
    }
};

class AsyncWriteBlockTest : public ::testing::Test {
   protected:
    int total_size;
    PartitionedFile *pf;
    int total_embeddings;
    int num_per_evict;
    int num_partitions;
    int partition_size;
    int embedding_size;
    int dtype_size;
    torch::Dtype dtype;
    string filename;
    int fd;
    torch::Tensor rand_tensor_float32;
    std::vector<Partition *> partitions;

    AsyncWriteBlockTest() {
        total_embeddings = 45;
        num_partitions = 5;
        partition_size = 10;
        embedding_size = 10000;
        dtype = torch::kFloat32;
        dtype_size = get_dtype_size_wrapper(dtype);
        total_size = partition_size * embedding_size * dtype_size;
        num_per_evict = 5;
    }

    ~AsyncWriteBlockTest() {}

    void SetUp() override {
        filename = testing::TempDir() + "asyn_write_block.txt";
        fd = createTmpFile(filename);
        ASSERT_NE(fd, -1);
        ASSERT_EQ(genRandTensorAndWriteToFile(rand_tensor_float32, total_embeddings, embedding_size, dtype, fd),
                  total_embeddings * embedding_size * dtype_size);
        pf = new PartitionedFile(filename, num_partitions, partition_size, embedding_size, total_embeddings, dtype);
    }

    void TearDown() override {
        remove(filename.c_str());
        delete (pf);
        close(fd);
        for (int i = 0; i < partitions.size(); i++) {
            delete (partitions[i]);
        }
    }
};

TEST_F(PartitionBufferTest, TestPartitionBufferOrdering) {
    initializePartitionBuffer(false);
    pb->unload(false);
    pb->load();
    std::vector<int> admits, evicts;
    for (int i = 2; i <= 4; i++) {
        tryNextSwapAndAssert(pb, std::vector<int>(1, i), std::vector<int>(1, i - 1))
    }
    tryNextSwapAndAssert(pb, std::vector<int>(1, 1), std::vector<int>(1, 0));
    for (int i = 4; i >= 3; i--) {
        tryNextSwapAndAssert(pb, std::vector<int>(1, i - 1), std::vector<int>(1, i));
    }
    tryNextSwapAndAssert(pb, std::vector<int>(1, 3), std::vector<int>(1, 1));
    tryNextSwapAndAssert(pb, std::vector<int>(1, 4), std::vector<int>(1, 3));
    tryNextSwapAndAssert(pb, std::vector<int>(1, 3), std::vector<int>(1, 2));
    ASSERT_EQ(pb->hasSwap(), false);
}

TEST_F(PartitionBufferTest, TestPartitionBufferPrefetch) {
    initializePartitionBuffer(true);
    std::vector<int> admits, evicts;
    for (int i = 2; i <= 4; i++) {
        tryNextSwapAndAssert(pb, std::vector<int>(1, i), std::vector<int>(1, i - 1))
    }
    tryNextSwapAndAssert(pb, std::vector<int>(1, 1), std::vector<int>(1, 0));
    for (int i = 4; i >= 3; i--) {
        tryNextSwapAndAssert(pb, std::vector<int>(1, i - 1), std::vector<int>(1, i));
    }
    tryNextSwapAndAssert(pb, std::vector<int>(1, 3), std::vector<int>(1, 1));
    tryNextSwapAndAssert(pb, std::vector<int>(1, 4), std::vector<int>(1, 3));
    tryNextSwapAndAssert(pb, std::vector<int>(1, 3), std::vector<int>(1, 2));
    ASSERT_EQ(pb->hasSwap(), false);
}

TEST_F(PartitionBufferTest, TestPartitionBufferIndexRead) {
    initializePartitionBuffer(false);
    torch::Tensor indices = pb->getRandomIds(20);
    torch::Tensor expected = rand_tensor_float32.index_select(0, indices);
    ASSERT_EQ(expected.equal(pb->indexRead(indices)), true);

    // indexRead should take in only 1d tensors.
    ASSERT_THROW(pb->indexRead(torch::randint(1000, {10, 10}, torch::kInt64)), std::runtime_error);
}

TEST_F(PartitionBufferTest, TestPartitionBufferIndexAdd) {
    initializePartitionBuffer(false);
    torch::Tensor indices = std::get<0>(at::_unique(pb->getRandomIds(1000)));
    torch::Tensor rand_values = torch::randint(1000, {indices.size(0), rows * cols}, torch::kFloat32);
    torch::Tensor updated_values = rand_tensor_float32.index_add_(0, indices, rand_values).index_select(0, indices);
    pb->indexAdd(indices, rand_values);
    ASSERT_EQ(updated_values.equal(pb->indexRead(indices)), true);

    // indexAdd should check tensor dims
    ASSERT_THROW(pb->indexAdd(indices, torch::randint(1000, {indices.size(0) + 1, rows * cols}, torch::kFloat32)), std::runtime_error);
    ASSERT_THROW(pb->indexAdd(indices, torch::randint(1000, {indices.size(0), rows * cols + 1}, torch::kFloat32)), std::runtime_error);
    ASSERT_THROW(pb->indexAdd(torch::randint(1000, {10, 10}, torch::kInt64), rand_values), std::runtime_error);
}

TEST_F(PartitionBufferTest, TestPartitionBufferSync) {
    initializePartitionBuffer(false);
    torch::Tensor rand;
    ASSERT_EQ(genRandTensorAndWriteToFile(rand, 2, embedding_size, dtype, fd), 2 * embedding_size * dtype_size);
    pb->unload(true);
    rand = torch::Tensor();
    rand = torch::randn({total_embeddings, embedding_size}, dtype);
    ASSERT_EQ(pread_wrapper(fd, (void *)rand.data_ptr(), total_embeddings * embedding_size * dtype_size, 0), total_embeddings * embedding_size * dtype_size);
    ASSERT_EQ(rand_tensor_float32.equal(rand), true);
}

TEST_F(PartitionBufferTest, TestPartitionBufferGlobalMap) {
    initializePartitionBuffer(false);
    torch::Tensor exp_map = -torch::ones({total_embeddings}, torch::kInt64);
    exp_map.slice(0, 0, 20) = torch::arange(0, 20);
    ASSERT_EQ(exp_map.equal(pb->getGlobalToLocalMap(true)), true);
    exp_map.slice(0, 10, 20) = -torch::ones({10}, torch::kInt64);
    exp_map.slice(0, 20, 30) = torch::arange(10, 20);
    ASSERT_EQ(exp_map.equal(pb->getGlobalToLocalMap(false)), true);
}

TEST_F(PartitionedFileTest, TestReadPartition) {
    int idx_offset = (num_partitions - 1) * partition_size;
    Partition p(num_partitions - 1, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                idx_offset * embedding_size * dtype_size);
    torch::Tensor rand_tensor = torch::randint(1000, {partition_size, embedding_size}, torch::kFloat32);
    void *addr = rand_tensor.data_ptr();
    pf->readPartition(addr, &p);
    torch::Tensor indices = at::randint(idx_offset, total_embeddings, 10, torch::kInt64);
    torch::Tensor returned_values = p.indexRead(indices);
    ASSERT_EQ(returned_values.equal(rand_tensor_float32.index_select(0, indices)), true);

    // check for exceptions on passing null ptrs.
    ASSERT_THROW(pf->readPartition(addr, NULL), std::runtime_error);
    ASSERT_THROW(pf->readPartition(NULL, &p), std::runtime_error);
}

TEST_F(PartitionedFileTest, TestWritePartition) {
    int idx_offset = (num_partitions - 1) * partition_size;
    Partition p(num_partitions - 1, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                idx_offset * embedding_size * dtype_size);
    void *addr = (void *)((char *)rand_tensor_float32.data_ptr() + idx_offset * embedding_size * dtype_size);
    pf->readPartition(addr, &p);

    // write random data to PartitionedFile
    torch::Tensor rand;
    ASSERT_EQ(genRandTensorAndWriteToFile(rand, total_embeddings, embedding_size, dtype, fd), total_embeddings * embedding_size * dtype_size);

    // write prev data
    pf->writePartition(&p, false);
    Partition p_(num_partitions - 1, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                 idx_offset * embedding_size * dtype_size);
    addr = (void *)((char *)rand.data_ptr() + idx_offset * embedding_size * dtype_size);
    pf->readPartition(addr, &p_);
    ASSERT_EQ(p.tensor_.equal(p_.tensor_), true);

    // writePartition after clearmem should throw an exception
    pf->writePartition(&p, true);
    ASSERT_THROW(pf->writePartition(&p, true), std::runtime_error);

    // writePartition should throw error on passing NULL;
    ASSERT_THROW(pf->writePartition(NULL, true), std::runtime_error);
}

TEST_F(LookaheadBlockTest, TestMoveToBuffer) {
    for (int i = 0; i < num_partitions; i++) {
        int idx_offset = i * partition_size;
        partitions.push_back(new Partition(i, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                                           idx_offset * embedding_size * dtype_size));
        returned_partitions.push_back(new Partition(i, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                                                    idx_offset * embedding_size * dtype_size));
    }

    LookaheadBlock lb(total_size, pf, num_per_lookahead);
    vector<Partition *> curr_partitions(2);
    for (int i = 0; i < 2; i++) curr_partitions[i] = partitions[i];
    lb.start(curr_partitions);

    vector<void *> buff_mem(num_per_lookahead);
    for (int i = 0; i < buff_mem.size(); i++) buff_mem[i] = malloc(partition_size * embedding_size * dtype_size);
    vector<int64_t> buff_ids;
    buff_ids.push_back(0);
    buff_ids.push_back(1);
    for (int i = 2; i < 4; i++) curr_partitions[i - 2] = partitions[i];
    lb.move_to_buffer(buff_mem, buff_ids, curr_partitions);
    for (int i = 0; i < 2; i++) {
        pf->readPartition(buff_mem[i], returned_partitions[i]);
        ASSERT_EQ(returned_partitions[i]->tensor_.equal(partitions[i]->tensor_), true);
    }

    curr_partitions.pop_back();
    curr_partitions[0] = partitions[4];
    lb.move_to_buffer(buff_mem, buff_ids, curr_partitions);
    for (int i = 2; i < 4; i++) {
        pf->readPartition(buff_mem[i - 2], returned_partitions[i]);
        ASSERT_EQ(returned_partitions[i]->tensor_.equal(partitions[i]->tensor_), true);
    }

    buff_ids.pop_back();
    curr_partitions.pop_back();

    // move_to_buffer should throw an exception when buffer size is less than the existing partitions
    ASSERT_THROW(lb.move_to_buffer(vector<void *>(), vector<int64_t>(), curr_partitions), std::runtime_error);

    lb.move_to_buffer(buff_mem, buff_ids, curr_partitions);
    pf->readPartition(buff_mem[0], returned_partitions[4]);
    ASSERT_EQ(returned_partitions[4]->tensor_.equal(partitions[4]->tensor_), true);
    lb.stop();
    for (int i = 0; i < buff_mem.size(); i++) free(buff_mem[i]);
}

TEST_F(AsyncWriteBlockTest, TestAsyncWrite) {
    for (int i = 0; i < num_partitions; i++) {
        int idx_offset = i * partition_size;
        partitions.push_back(new Partition(i, std::min(partition_size, total_embeddings - idx_offset), embedding_size, dtype, idx_offset,
                                           idx_offset * embedding_size * dtype_size));
    }

    LookaheadBlock lb(total_size, pf, num_per_evict);
    lb.start(partitions);
    AsyncWriteBlock awb(total_size, pf, num_per_evict);
    awb.start();

    vector<void *> buff_mem(num_per_evict);
    vector<int64_t> buff_ids(num_per_evict);
    for (int i = 0; i < buff_mem.size(); i++) {
        buff_mem[i] = malloc(partition_size * embedding_size * dtype_size);
        buff_ids[i] = i;
    }
    lb.move_to_buffer(buff_mem, buff_ids, vector<Partition *>());
    torch::Tensor rand_tensor = torch::randn({total_embeddings, embedding_size}, dtype);
    for (int i = 0; i < partitions.size(); i++) {
        int idx_offset = i * partition_size;
        void *addr = (void *)((char *)rand_tensor.data_ptr() + idx_offset * embedding_size * dtype_size);
        memcpy_wrapper(partitions[i]->data_ptr_, addr, partitions[i]->partition_size_ * embedding_size * dtype_size);
        partitions[i]->tensor_ = torch::from_blob(partitions[i]->data_ptr_, {partitions[i]->partition_size_, embedding_size}, dtype);
    }

    awb.async_write(partitions);
    // wait until the write happens
    std::unique_lock lock(*(awb.lock_));
    awb.cv_.wait(lock, [&awb] { return awb.present_ == false; });
    lock.unlock();
    awb.cv_.notify_all();

    pread_wrapper(fd, (void *)rand_tensor_float32.data_ptr(), total_embeddings * embedding_size * dtype_size, 0);
    ASSERT_EQ(rand_tensor_float32.equal(rand_tensor), true);

    // async_write should throw an exception when the passed in partitions has a length greater than the mem block
    partitions.push_back(partitions[0]);
    ASSERT_THROW(awb.async_write(partitions), std::runtime_error);
    partitions.pop_back();

    for (int i = 0; i < buff_mem.size(); i++) free(buff_mem[i]);
    lb.stop();
    awb.stop();
}