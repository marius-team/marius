#include <fcntl.h>
#include <unistd.h>

#include <memory>

#include "configuration/config.h"
#include "gtest/gtest.h"
#include "storage/storage.h"
#include "testing_util.h"

#define tryNextSwapAndAssert(pbs, admits_, evicts_) \
    ASSERT_EQ(pbs.hasSwap(), true);                 \
    admits = pbs.getNextAdmit();                    \
    evicts = pbs.getNextEvict();                    \
    ASSERT_EQ(admits, admits_);                     \
    ASSERT_EQ(evicts, evicts_);                     \
    pbs.performNextSwap();

class StorageTest : public ::testing::Test {
   protected:
    vector<std::string> filenames_array;
    vector<int> fd_array;
    int64_t dim0_size;
    int64_t dim1_size;
    vector<torch::Dtype> dtype_array;
    vector<int> dtype_size_array;
    vector<torch::Tensor> rand_tensors_array;
    std::string edges_weight_path;
    int weight_file_fd;

    StorageTest() {
        dim0_size = 46;
        dim1_size = 1000;
        dtype_array = {torch::kInt32, torch::kInt64, torch::kFloat16, torch::kFloat32, torch::kFloat64};
        for (int i = 0; i < dtype_array.size(); i++) dtype_size_array.push_back(get_dtype_size_wrapper(dtype_array[i]));
    }

    ~StorageTest() {}

    void SetUp() override {
        for (int i = 0; i < dtype_array.size(); i++) {
            filenames_array.push_back(testing::TempDir() + "storage_" + std::to_string(i) + ".txt");
            fd_array.push_back(createTmpFile(filenames_array[i]));
            ASSERT_NE(fd_array[i], -1);
            torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);
            rand_tensors_array.push_back(rand_tensor);
        }

        edges_weight_path = testing::TempDir() + "flat_file_edge_weights.txt";
        weight_file_fd = createTmpFile(edges_weight_path);
    }

    void TearDown() override {
        for (int i = 0; i < dtype_array.size(); i++) {
            close(fd_array[i]);
            remove(filenames_array[i].c_str());
        }

        close(weight_file_fd);
        remove(edges_weight_path.c_str());
    }
};

class FlatFileTest : public StorageTest {
   protected:
    int num_edges;
    int num_cols;
    int num_nodes;
    std::string edges_data_path;
    std::string edges_bucket_partition_path;

    FlatFileTest() {
        num_edges = 1000;
        num_cols = 3;
        num_nodes = 100;
        edges_data_path = testing::TempDir() + "flat_file_edges.bin";
        edges_bucket_partition_path = testing::TempDir() + "flat_file_edge_partitions.txt";
    }

    ~FlatFileTest() {}
};

class InMemoryTest : public StorageTest {
   protected:
    int num_edges;
    int num_cols;
    int num_nodes;
    std::string edges_data_path;
    std::string edges_bucket_partition_path;
    InMemoryTest() {
        num_edges = 1000;
        num_cols = 3;
        num_nodes = 100;
        edges_data_path = testing::TempDir() + "in_memory_edges.bin";
        edges_bucket_partition_path = testing::TempDir() + "in_memory_edge_partitions.txt";
    }

    ~InMemoryTest() {}
};

class PartitionBufferStorageTest : public StorageTest {
   protected:
    shared_ptr<PartitionBufferOptions> options;
    int capacity;
    int num_partitions;
    int64_t partition_size;
    int fine_to_coarse_ratio;
    vector<torch::Tensor> buffer_states;

    PartitionBufferStorageTest() {
        shared_ptr<MariusConfig> p = loadConfig(std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/fb15k_237.yaml");
        options = std::dynamic_pointer_cast<PartitionBufferOptions>(p->storage->embeddings->options);

        capacity = 2;
        num_partitions = 5;
        partition_size = 10;
        fine_to_coarse_ratio = 2;

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

    ~PartitionBufferStorageTest() {}
};

TEST_F(FlatFileTest, TestFlatFileWrite) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        ASSERT_EQ(pread_wrapper(fd_array[i], (void *)rand_tensor.data_ptr(), dim0_size * dim1_size * dtype_size_array[i], 0),
                  dim0_size * dim1_size * dtype_size_array[i]);
        ASSERT_EQ(rand_tensors_array[i].equal(rand_tensor), true);

        flat_file.load();
        flat_file.load();
        flat_file.unload(true);
        ASSERT_THROW(flat_file.rangePut(17, rand_tensor.index_select(0, torch::arange(17, 37))), std::runtime_error);

        flat_file.load();
        torch::Tensor rand_sub_tensor = getRandTensor(20, dim1_size, dtype_array[i]);
        rand_tensor.slice(0, 17, 37) = rand_sub_tensor;
        flat_file.rangePut(17, rand_tensor.index_select(0, torch::arange(17, 37)));
        ASSERT_EQ(rand_tensor.index_select(0, torch::arange(17, 37)).equal(flat_file.range(17, 20)), true);

        ASSERT_THROW(flat_file.rangePut(37, rand_tensors_array[i].index_select(0, torch::arange(17, 37))), std::runtime_error);
        ASSERT_THROW(flat_file.range(17, 30), std::runtime_error);
        ASSERT_THROW(flat_file.rangePut(17, torch::Tensor()), std::runtime_error);
    }
}

TEST_F(FlatFileTest, TestFlatFileCopy) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);

        std::string new_filename = testing::TempDir() + "new_storage.txt";
        int new_fd = createTmpFile(new_filename);
        torch::Tensor rand_tensor;
        ASSERT_EQ(genRandTensorAndWriteToFile(rand_tensor, dim0_size, dim1_size, dtype_array[i], new_fd), dim0_size * dim1_size * dtype_size_array[i]);
        ASSERT_NE(new_fd, -1);

        flat_file.copy(new_filename, false);
        ASSERT_EQ(pread_wrapper(new_fd, (void *)rand_tensor.data_ptr(), dim0_size * dim1_size * dtype_size_array[i], 0),
                  dim0_size * dim1_size * dtype_size_array[i]);
        ASSERT_EQ(rand_tensors_array[i].equal(rand_tensor), true);

        close(new_fd);
        remove(new_filename.c_str());
    }
}

TEST_F(FlatFileTest, TestFlatFileShuffle) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        rand_tensor.slice(0, 0, dim0_size) = rand_tensors_array[i].slice(0, 0, dim0_size);
        flat_file.shuffle();
        ASSERT_THROW(flat_file.range(0, dim0_size), std::runtime_error);

        flat_file.load();
        rand_tensors_array[i] = flat_file.range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(rand_tensors_array[i], rand_tensor), true);
    }
}

TEST_F(FlatFileTest, TestFlatFileShuffleWithWeights) {
    // Create the initial weight file
    shared_ptr<Storage> weight_file = std::make_shared<FlatFile>(edges_weight_path, dim0_size, 1, torch::kFloat32);
    torch::Tensor weight_tensor = getRandTensor(dim0_size, 1, torch::kFloat32);
    weight_file->load();

    for (int i = 0; i < dtype_size_array.size(); i++) {
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        weight_file->rangePut(0, weight_tensor);
        rand_tensor.slice(0, 0, dim0_size) = rand_tensors_array[i].slice(0, 0, dim0_size);
        flat_file.shuffle(weight_file);

        flat_file.load();
        rand_tensors_array[i] = flat_file.range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(rand_tensors_array[i], rand_tensor), true);
        torch::Tensor sorted_weight_tensor = weight_file->range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(sorted_weight_tensor, weight_tensor), true);
    }
}

TEST_F(FlatFileTest, TestFlatFileShuffleEdges) {
    // Create and populate the flat file
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);
    FlatFile flat_file(edges_data_path, 0, num_cols, torch::kInt32);
    flat_file.append(rand_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }

    // Shuffle the flat file
    flat_file.readPartitionSizes(edges_bucket_partition_path);
    flat_file.shuffle();
    flat_file.load();
    torch::Tensor shuffled_tensor = flat_file.range(0, num_edges);
    vector<int64_t> edge_bucket_sizes = flat_file.getEdgeBucketSizes();
    ASSERT_EQ(checkPermWithBuckets(rand_tensor, shuffled_tensor, edge_bucket_sizes), true);
}

TEST_F(FlatFileTest, TestFlatFileShuffleEdgesWithWeights) {
    // Create and populate the flat file
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);
    FlatFile flat_file(edges_data_path, 0, num_cols, torch::kInt32);
    flat_file.append(rand_tensor);

    // Created weights file
    shared_ptr<Storage> weight_file = std::make_shared<FlatFile>(edges_weight_path, num_edges, 1, torch::kFloat32);
    torch::Tensor weight_tensor = getRandTensor(num_edges, 1, torch::kFloat32);
    weight_file->load();
    weight_file->rangePut(0, weight_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }

    // Shuffle the flat file
    flat_file.readPartitionSizes(edges_bucket_partition_path);
    flat_file.shuffle(weight_file);
    flat_file.load();
    torch::Tensor shuffled_tensor = flat_file.range(0, num_edges);
    vector<int64_t> edge_bucket_sizes = flat_file.getEdgeBucketSizes();
    torch::Tensor shuffled_weight_tensor = weight_file->range(0, num_edges);
    ASSERT_EQ(checkPermWithBuckets(rand_tensor, shuffled_tensor, edge_bucket_sizes), true);
    ASSERT_EQ(checkPermWithBuckets(weight_tensor, shuffled_weight_tensor, edge_bucket_sizes), true);
}

TEST_F(FlatFileTest, TestFlatFileSort) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        rand_tensor.slice(0, 0, dim0_size) = rand_tensors_array[i].slice(0, 0, dim0_size);
        flat_file.sort(true);
        ASSERT_THROW(flat_file.range(0, dim0_size), std::runtime_error);

        flat_file.load();
        rand_tensors_array[i] = flat_file.range(0, dim0_size);
        rand_tensor.copy_(rand_tensor.index_select(0, torch::argsort(rand_tensor.select(1, 0))));
        ASSERT_EQ(rand_tensor.equal(rand_tensors_array[i]), true);
    }
}

TEST_F(FlatFileTest, TestFlatFileWeightSort) {
    // Create the initial weight file
    shared_ptr<Storage> weight_file = std::make_shared<FlatFile>(edges_weight_path, dim0_size, 1, torch::kFloat32);
    torch::Tensor weight_tensor = getRandTensor(dim0_size, 1, torch::kFloat32);
    weight_file->load();

    for (int i = 0; i < dtype_size_array.size(); i++) {
        // Create the file
        FlatFile flat_file(filenames_array[i], 0, dim1_size, dtype_array[i]);
        flat_file.append(rand_tensors_array[i]);

        // Sort using the src column and verify that it matches expected
        weight_file->rangePut(0, weight_tensor);
        flat_file.sort(true, weight_file);
        flat_file.load();
        torch::Tensor actual_src_sort = flat_file.range(0, dim0_size);
        torch::Tensor actual_src_weights = weight_file->range(0, dim0_size);
        torch::Tensor expected_src_sort = rand_tensors_array[i].slice(0, 0, dim0_size);
        torch::Tensor expected_src_weights = weight_tensor.slice(0, 0, dim0_size);
        torch::Tensor src_sort_order = torch::argsort(expected_src_sort.select(1, 0));
        expected_src_sort.copy_(expected_src_sort.index_select(0, src_sort_order));
        expected_src_weights.copy_(expected_src_weights.index_select(0, src_sort_order));
        ASSERT_EQ(expected_src_sort.equal(actual_src_sort), true);
        ASSERT_EQ(expected_src_weights.equal(actual_src_weights), true);

        // Now sort using the last column and make sure that produces the same result
        weight_file->rangePut(0, weight_tensor);
        flat_file.sort(false, weight_file);
        flat_file.load();
        torch::Tensor actual_dst_sort = flat_file.range(0, dim0_size);
        torch::Tensor actual_dst_weights = weight_file->range(0, dim0_size);
        torch::Tensor expected_dst_sort = rand_tensors_array[i].slice(0, 0, dim0_size);
        torch::Tensor expected_dst_weights = weight_tensor.slice(0, 0, dim0_size);
        torch::Tensor dst_sort_order = torch::argsort(expected_dst_sort.select(1, -1));
        expected_dst_sort.copy_(expected_dst_sort.index_select(0, dst_sort_order));
        expected_dst_weights.copy_(expected_dst_weights.index_select(0, dst_sort_order));
        ASSERT_EQ(expected_dst_sort.equal(actual_dst_sort), true);
        ASSERT_EQ(expected_dst_weights.equal(actual_dst_weights), true);
    }
}

TEST_F(FlatFileTest, TestFlatFileSortEdges) {
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);

    FlatFile flat_file(edges_data_path, 0, num_cols, torch::kInt32);
    flat_file.append(rand_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }
    flat_file.readPartitionSizes(edges_bucket_partition_path);
    flat_file.load();

    torch::Tensor rand_tensor_1 = getRandTensor(num_edges, num_cols, torch::kInt32);
    torch::Tensor rand_tensor_2 = getRandTensor(num_edges, num_cols, torch::kInt32);
    rand_tensor_1.copy_(flat_file.range(0, num_edges));
    rand_tensor_2.copy_(rand_tensor_1);

    flat_file.sort(true);
    vector<int64_t> edge_bucket_sizes = flat_file.getEdgeBucketSizes();
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes);
    rand_tensor_1 = flat_file.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);

    flat_file.sort(false);
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, -1);
    rand_tensor_1 = flat_file.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);

    remove(edges_bucket_partition_path.c_str());
    remove(edges_data_path.c_str());
}

TEST_F(FlatFileTest, TestFlatFileSortEdgesWithWeight) {
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);

    FlatFile flat_file(edges_data_path, 0, num_cols, torch::kInt32);
    flat_file.append(rand_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }
    flat_file.readPartitionSizes(edges_bucket_partition_path);
    flat_file.load();

    // Create the weight file
    shared_ptr<Storage> weight_file = std::make_shared<FlatFile>(edges_weight_path, num_edges, 1, torch::kFloat32);
    torch::Tensor weight_tensor = getRandTensor(num_edges, 1, torch::kFloat32);
    weight_file->load();

    torch::Tensor rand_tensor_1 = getRandTensor(num_edges, num_cols, torch::kInt32);
    torch::Tensor rand_tensor_2 = getRandTensor(num_edges, num_cols, torch::kInt32);
    rand_tensor_1.copy_(flat_file.range(0, num_edges));
    rand_tensor_2.copy_(rand_tensor_1);

    // Sort using the src column
    weight_file->rangePut(0, weight_tensor);
    flat_file.sort(true, weight_file);
    torch::Tensor expected_src_weight = weight_tensor.slice(0, 0, num_edges);
    vector<int64_t> edge_bucket_sizes = flat_file.getEdgeBucketSizes();
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, 0, &expected_src_weight);
    rand_tensor_1 = flat_file.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);
    torch::Tensor actual_src_weight = weight_file->range(0, num_edges);
    ASSERT_EQ(expected_src_weight.equal(actual_src_weight), true);

    // Sort using the dst column
    weight_file->rangePut(0, weight_tensor);
    flat_file.sort(false, weight_file);
    torch::Tensor expected_dst_weight = weight_tensor.slice(0, 0, num_edges);
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, -1, &expected_dst_weight);
    rand_tensor_1 = flat_file.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);
    torch::Tensor actual_dst_weight = weight_file->range(0, num_edges);
    ASSERT_EQ(expected_dst_weight.equal(actual_dst_weight), true);

    remove(edges_bucket_partition_path.c_str());
    remove(edges_data_path.c_str());
}

TEST_F(InMemoryTest, TestIndexRead) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        InMemory in_memory(filenames_array[i], rand_tensors_array[i], torch::kCPU);
        in_memory.load();
        torch::Tensor rand_tensor = getRandTensor(10, dim1_size, dtype_array[i]);
        ASSERT_EQ(pread_wrapper(fd_array[i], rand_tensor.data_ptr(), 10 * dim1_size * dtype_size_array[i], 10 * dim1_size * dtype_size_array[i]),
                  10 * dim1_size * dtype_size_array[i]);
        ASSERT_EQ(in_memory.indexRead(torch::arange(10, 20)).equal(rand_tensor), true);

        ASSERT_THROW(in_memory.indexRead(torch::randint(100, {10, 10}, dtype_array[i])), std::runtime_error);
        ASSERT_THROW(in_memory.range(17, 30), std::runtime_error);
    }
}

TEST_F(InMemoryTest, TestIndexAdd) {
    // just iterate over torch::kFloat32, indexAdd doesn't support any other dtype
    for (int i = 3; i < 4; i++) {
        InMemory in_memory(filenames_array[i], rand_tensors_array[i], torch::kCPU);
        in_memory.load();
        torch::Tensor indices = torch::arange(0, dim0_size);
        torch::Tensor rand_values = torch::randint(1000, {indices.size(0), dim1_size}, dtype_array[i]);
        torch::Tensor updated_values = rand_tensors_array[i].index_add_(0, indices, rand_values).index_select(0, indices);
        in_memory.indexAdd(indices, rand_values);
        ASSERT_EQ(updated_values.equal(in_memory.indexRead(indices)), true);

        // indexAdd should check tensor dims
        ASSERT_THROW(in_memory.indexAdd(indices, torch::randn({indices.size(0) + 1, dim1_size}, dtype_array[i])), std::runtime_error);
        ASSERT_THROW(in_memory.indexAdd(indices, torch::randn({indices.size(0), dim1_size + 1}, dtype_array[i])), std::runtime_error);
        ASSERT_THROW(in_memory.indexAdd(torch::randint(1000, {10, 10}, torch::kInt64), rand_values), std::runtime_error);
        rand_values = torch::Tensor();
        ASSERT_THROW(in_memory.indexAdd(indices, rand_values), std::runtime_error);
    }
}

TEST_F(InMemoryTest, TestIndexPut) {
    // just iterate over torch::kFloat32, indexPut doesn't support any other dtype
    for (int i = 3; i < 4; i++) {
        InMemory in_memory(filenames_array[i], rand_tensors_array[i], torch::kCPU);
        in_memory.load();
        torch::Tensor indices = std::get<0>(at::_unique(torch::randint(dim0_size, dim1_size, torch::kInt64)));
        torch::Tensor rand_values = getRandTensor(indices.size(0), dim1_size, dtype_array[i]);

        in_memory.indexPut(indices, rand_values);
        ASSERT_EQ(rand_values.equal(in_memory.indexRead(indices)), true);

        // indexPut should check tensor dims
        rand_values = getRandTensor(indices.size(0) + 1, dim1_size, dtype_array[i]);
        ASSERT_THROW(in_memory.indexPut(indices, rand_values), std::runtime_error);
        rand_values = getRandTensor(indices.size(0), dim1_size + 1, dtype_array[i]);
        ASSERT_THROW(in_memory.indexPut(indices, rand_values), std::runtime_error);
        rand_values = getRandTensor(indices.size(0), dim1_size, dtype_array[i]);
        ASSERT_THROW(in_memory.indexPut(torch::randint(1000, {10, 10}, torch::kInt64), rand_values), std::runtime_error);

        rand_values = torch::Tensor();
        ASSERT_THROW(in_memory.indexPut(indices, rand_values), std::runtime_error);
    }
}

TEST_F(InMemoryTest, TestInMemoryShuffle) {
    for (int i = 0; i < dtype_array.size(); i++) {
        InMemory in_memory(filenames_array[i], rand_tensors_array[i], torch::kCPU);
        in_memory.load();
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        rand_tensor.slice(0, 0, dim0_size) = rand_tensors_array[i].slice(0, 0, dim0_size);
        in_memory.shuffle();
        rand_tensors_array[i] = in_memory.range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(rand_tensors_array[i], rand_tensor), true);
    }
}

TEST_F(InMemoryTest, TestInMemoryShuffleWithWeights) {
    // Create the initial weight file
    shared_ptr<Storage> weight_file = std::make_shared<InMemory>(edges_weight_path, dim0_size, 1, torch::kFloat32, torch::kCPU);
    torch::Tensor weight_tensor = getRandTensor(dim0_size, 1, torch::kFloat32);
    weight_file->load();

    for (int i = 0; i < dtype_size_array.size(); i++) {
        // Created in memory file
        InMemory in_memory(filenames_array[i], dim0_size, dim1_size, dtype_array[i], torch::kCPU);
        torch::Tensor rand_tensor = rand_tensors_array[i].slice(0, 0, dim0_size);
        in_memory.load();
        in_memory.rangePut(0, rand_tensor);

        // Shuffle the file
        weight_file->rangePut(0, weight_tensor);
        in_memory.shuffle(weight_file);

        // Verify the permutation
        rand_tensors_array[i] = in_memory.range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(rand_tensors_array[i], rand_tensor), true);
        torch::Tensor sorted_weight_tensor = weight_file->range(0, dim0_size);
        ASSERT_EQ(checkPermOf2dTensor(sorted_weight_tensor, weight_tensor), true);
    }
}

TEST_F(InMemoryTest, TestInMemoryShuffleEdges) {
    // Create and populate the in memory storage
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);
    InMemory in_memory(edges_data_path, num_edges, num_cols, torch::kInt32, torch::kCPU);
    in_memory.load();
    in_memory.rangePut(0, rand_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }

    // Shuffle the file
    in_memory.readPartitionSizes(edges_bucket_partition_path);
    in_memory.shuffle();
    torch::Tensor shuffled_tensor = in_memory.range(0, num_edges);
    vector<int64_t> edge_bucket_sizes = in_memory.getEdgeBucketSizes();
    ASSERT_EQ(checkPermWithBuckets(rand_tensor, shuffled_tensor, edge_bucket_sizes), true);
}

TEST_F(InMemoryTest, TestInMemoryShuffleEdgesWithWeights) {
    // Create and populate the in memory storage
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);
    InMemory in_memory(edges_data_path, num_edges, num_cols, torch::kInt32, torch::kCPU);
    in_memory.load();
    in_memory.rangePut(0, rand_tensor);

    // Created weights file
    shared_ptr<Storage> weight_file = std::make_shared<InMemory>(edges_weight_path, num_edges, 1, torch::kFloat32, torch::kCPU);
    torch::Tensor weight_tensor = getRandTensor(num_edges, 1, torch::kFloat32);
    weight_file->load();
    weight_file->rangePut(0, weight_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }

    // Shuffle the file with weights
    in_memory.readPartitionSizes(edges_bucket_partition_path);
    in_memory.shuffle(weight_file);
    torch::Tensor shuffled_tensor = in_memory.range(0, num_edges);
    vector<int64_t> edge_bucket_sizes = in_memory.getEdgeBucketSizes();
    torch::Tensor shuffled_weight_tensor = weight_file->range(0, num_edges);
    ASSERT_EQ(checkPermWithBuckets(rand_tensor, shuffled_tensor, edge_bucket_sizes), true);
    ASSERT_EQ(checkPermWithBuckets(weight_tensor, shuffled_weight_tensor, edge_bucket_sizes), true);
}

TEST_F(InMemoryTest, TestInMemorySort) {
    for (int i = 0; i < dtype_array.size(); i++) {
        InMemory in_memory(filenames_array[i], rand_tensors_array[i], torch::kCPU);
        in_memory.load();
        torch::Tensor rand_tensor = getRandTensor(dim0_size, dim1_size, dtype_array[i]);

        rand_tensor.slice(0, 0, dim0_size) = rand_tensors_array[i].slice(0, 0, dim0_size);
        in_memory.sort(true);
        rand_tensors_array[i] = in_memory.range(0, dim0_size);
        rand_tensor.copy_(rand_tensor.index_select(0, torch::argsort(rand_tensor.select(1, 0))));
        ASSERT_EQ(rand_tensor.equal(rand_tensors_array[i]), true);
    }
}

TEST_F(InMemoryTest, TestInMemoryWeightSort) {
    // Create the initial weight file
    shared_ptr<Storage> weight_file = std::make_shared<InMemory>(edges_weight_path, dim0_size, 1, torch::kFloat32, torch::kCPU);
    torch::Tensor weight_tensor = getRandTensor(dim0_size, 1, torch::kFloat32);
    weight_file->load();

    for (int i = 0; i < dtype_size_array.size(); i++) {
        // Create the file
        InMemory in_memory(filenames_array[i], dim0_size, dim1_size, dtype_array[i], torch::kCPU);
        in_memory.load();
        in_memory.rangePut(0, rand_tensors_array[i]);

        // Sort using the src column and verify that it matches expected
        weight_file->rangePut(0, weight_tensor);
        in_memory.sort(true, weight_file);
        torch::Tensor actual_src_sort = in_memory.range(0, dim0_size);
        torch::Tensor actual_src_weights = weight_file->range(0, dim0_size);
        torch::Tensor expected_src_sort = rand_tensors_array[i].slice(0, 0, dim0_size);
        torch::Tensor expected_src_weights = weight_tensor.slice(0, 0, dim0_size);
        torch::Tensor src_sort_order = torch::argsort(expected_src_sort.select(1, 0));
        expected_src_sort.copy_(expected_src_sort.index_select(0, src_sort_order));
        expected_src_weights.copy_(expected_src_weights.index_select(0, src_sort_order));
        ASSERT_EQ(expected_src_sort.equal(actual_src_sort), true);
        ASSERT_EQ(expected_src_weights.equal(actual_src_weights), true);

        // Now sort using the last column and make sure that produces the same result
        weight_file->rangePut(0, weight_tensor);
        in_memory.sort(false, weight_file);
        torch::Tensor actual_dst_sort = in_memory.range(0, dim0_size);
        torch::Tensor actual_dst_weights = weight_file->range(0, dim0_size);
        torch::Tensor expected_dst_sort = rand_tensors_array[i].slice(0, 0, dim0_size);
        torch::Tensor expected_dst_weights = weight_tensor.slice(0, 0, dim0_size);
        torch::Tensor dst_sort_order = torch::argsort(expected_dst_sort.select(1, -1));
        expected_dst_sort.copy_(expected_dst_sort.index_select(0, dst_sort_order));
        expected_dst_weights.copy_(expected_dst_weights.index_select(0, dst_sort_order));
        ASSERT_EQ(expected_dst_sort.equal(actual_dst_sort), true);
        ASSERT_EQ(expected_dst_weights.equal(actual_dst_weights), true);
    }
}

TEST_F(InMemoryTest, TestInMemorySortEdges) {
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);

    InMemory in_memory(edges_data_path, num_edges, num_cols, torch::kInt32, torch::kCPU);
    in_memory.load();
    in_memory.rangePut(0, rand_tensor);
    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }
    in_memory.readPartitionSizes(edges_bucket_partition_path);
    in_memory.load();

    torch::Tensor rand_tensor_1 = getRandTensor(num_edges, num_cols, torch::kInt32, 10000);
    torch::Tensor rand_tensor_2 = getRandTensor(num_edges, num_cols, torch::kInt32);
    rand_tensor_1.copy_(in_memory.range(0, num_edges));
    rand_tensor_2.copy_(rand_tensor_1);

    in_memory.sort(true);
    vector<int64_t> edge_bucket_sizes = in_memory.getEdgeBucketSizes();
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes);
    rand_tensor_1 = in_memory.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);

    in_memory.sort(false);
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, -1);
    rand_tensor_1 = in_memory.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);

    remove(edges_bucket_partition_path.c_str());
    remove(edges_data_path.c_str());
}

TEST_F(InMemoryTest, TestInMemorySortEdgesWithWeight) {
    torch::Tensor rand_tensor = getRandTensor(num_edges, num_cols, torch::kInt32, num_nodes);
    createTmpFile(edges_data_path);

    InMemory in_memory(edges_data_path, num_edges, num_cols, torch::kInt32, torch::kCPU);
    in_memory.load();
    in_memory.rangePut(0, rand_tensor);

    vector<int64_t> partition_sizes = partitionEdges(rand_tensor, 3, num_nodes + 1);
    createTmpFile(edges_bucket_partition_path);
    {
        std::ofstream ostrm;
        ostrm.open(edges_bucket_partition_path, std::ios::out | std::ios::trunc);
        for (int i = 0; i < partition_sizes.size(); i++) ostrm << partition_sizes[i] << "\n";
        ostrm.close();
    }
    in_memory.readPartitionSizes(edges_bucket_partition_path);

    // Create the weight file
    shared_ptr<Storage> weight_file = std::make_shared<FlatFile>(edges_weight_path, num_edges, 1, torch::kFloat32);
    torch::Tensor weight_tensor = getRandTensor(num_edges, 1, torch::kFloat32);
    weight_file->load();

    torch::Tensor rand_tensor_1 = getRandTensor(num_edges, num_cols, torch::kInt32);
    torch::Tensor rand_tensor_2 = getRandTensor(num_edges, num_cols, torch::kInt32);
    rand_tensor_1.copy_(in_memory.range(0, num_edges));
    rand_tensor_2.copy_(rand_tensor_1);

    // Sort using the src column
    weight_file->rangePut(0, weight_tensor);
    in_memory.sort(true, weight_file);
    torch::Tensor expected_src_weight = weight_tensor.slice(0, 0, num_edges);
    vector<int64_t> edge_bucket_sizes = in_memory.getEdgeBucketSizes();
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, 0, &expected_src_weight);
    rand_tensor_1 = in_memory.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);
    torch::Tensor actual_src_weight = weight_file->range(0, num_edges);
    ASSERT_EQ(expected_src_weight.equal(actual_src_weight), true);

    // Sort using the dst column
    weight_file->rangePut(0, weight_tensor);
    in_memory.sort(false, weight_file);
    torch::Tensor expected_dst_weight = weight_tensor.slice(0, 0, num_edges);
    sortWithinEdgeBuckets(rand_tensor_2, edge_bucket_sizes, -1, &expected_dst_weight);
    rand_tensor_1 = in_memory.range(0, num_edges);
    ASSERT_EQ(rand_tensor_1.equal(rand_tensor_2), true);
    torch::Tensor actual_dst_weight = weight_file->range(0, num_edges);
    ASSERT_EQ(expected_dst_weight.equal(actual_dst_weight), true);

    remove(edges_bucket_partition_path.c_str());
    remove(edges_data_path.c_str());
}

TEST_F(PartitionBufferStorageTest, TestBufferOrdering) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        PartitionBufferStorage pbs(filenames_array[i], rand_tensors_array[i], options);
        pbs.setBufferOrdering(buffer_states);
        pbs.load();

        std::vector<int> admits, evicts;
        for (int i = 2; i <= 4; i++) {
            tryNextSwapAndAssert(pbs, std::vector<int>(1, i), std::vector<int>(1, i - 1))
        }
        tryNextSwapAndAssert(pbs, std::vector<int>(1, 1), std::vector<int>(1, 0));
        for (int i = 4; i >= 3; i--) {
            tryNextSwapAndAssert(pbs, std::vector<int>(1, i - 1), std::vector<int>(1, i));
        }
        tryNextSwapAndAssert(pbs, std::vector<int>(1, 3), std::vector<int>(1, 1));
        tryNextSwapAndAssert(pbs, std::vector<int>(1, 4), std::vector<int>(1, 3));
        tryNextSwapAndAssert(pbs, std::vector<int>(1, 3), std::vector<int>(1, 2));
        ASSERT_EQ(pbs.hasSwap(), false);
    }
}

TEST_F(PartitionBufferStorageTest, TestIndexRead) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        PartitionBufferStorage pbs(filenames_array[i], rand_tensors_array[i], options);
        pbs.setBufferOrdering(buffer_states);
        pbs.load();

        torch::Tensor indices = pbs.getRandomIds(20);
        torch::Tensor expected = rand_tensors_array[i].index_select(0, indices);
        ASSERT_EQ(expected.equal(pbs.indexRead(indices)), true);

        // indexRead should take in only 1d tensors.
        ASSERT_THROW(pbs.indexRead(torch::randint(1000, {10, 10}, torch::kInt64)), std::runtime_error);
    }
}

TEST_F(PartitionBufferStorageTest, TestRangePut) {
    for (int i = 0; i < dtype_size_array.size(); i++) {
        PartitionBufferStorage pbs(filenames_array[i], rand_tensors_array[i], options);
        pbs.setBufferOrdering(buffer_states);
        pbs.load();

        torch::Tensor rand_sub_tensor = getRandTensor(20, dim1_size, dtype_array[i]);
        pbs.rangePut(0, rand_sub_tensor);
        pbs.unload(false);
        pbs.load();
        ASSERT_EQ(rand_sub_tensor.equal(pbs.indexRead(torch::arange(0, 20))), true);

        pbs.performNextSwap();
        rand_sub_tensor = getRandTensor(10, dim1_size, dtype_array[i]);
        pbs.rangePut(20, rand_sub_tensor);
        pbs.unload(false);
        pbs.setBufferOrdering(buffer_states);
        pbs.load();
        pbs.performNextSwap();
        ASSERT_EQ(rand_sub_tensor.equal(pbs.indexRead(torch::arange(10, 20))), true);
    }
}

TEST_F(PartitionBufferStorageTest, TestIndexAdd) {
    // just iterate over torch::kFloat32, indexAdd doesn't support any other dtype
    for (int i = 3; i < 4; i++) {
        PartitionBufferStorage pbs(filenames_array[i], rand_tensors_array[i], options);
        pbs.setBufferOrdering(buffer_states);
        pbs.load();

        torch::Tensor indices = std::get<0>(at::_unique(pbs.getRandomIds(20)));
        torch::Tensor rand_values = getRandTensor(indices.size(0), dim1_size, dtype_array[i]);
        torch::Tensor updated_values = rand_tensors_array[i].index_add_(0, indices, rand_values).index_select(0, indices);
        pbs.indexAdd(indices, rand_values);
        ASSERT_EQ(updated_values.equal(pbs.indexRead(indices)), true);

        ASSERT_EQ(pbs.getNumInMemory(), 20);

        // indexAdd should check tensor dims
        rand_values = getRandTensor(indices.size(0) + 1, dim1_size, dtype_array[i]);
        ASSERT_THROW(pbs.indexAdd(indices, rand_values), std::runtime_error);
        rand_values = getRandTensor(indices.size(0), dim1_size + 1, dtype_array[i]);
        ASSERT_THROW(pbs.indexAdd(indices, rand_values), std::runtime_error);
        rand_values = getRandTensor(indices.size(0), dim1_size, dtype_array[i]);
        ASSERT_THROW(pbs.indexAdd(torch::randint(1000, {10, 10}, torch::kInt64), rand_values), std::runtime_error);
        rand_values = torch::Tensor();
        ASSERT_THROW(pbs.indexAdd(indices, rand_values), std::runtime_error);
    }
}