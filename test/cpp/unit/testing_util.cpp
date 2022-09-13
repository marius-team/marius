#include "testing_util.h"

#include "gtest/gtest.h"
#include "util.h"

int createTmpFile(std::string &filename) { return open(filename.c_str(), O_RDWR | O_CREAT, 0777); }

torch::Tensor getRandTensor(int dim0_size, int dim1_size, torch::Dtype dtype, int max_val) {
    if (dtype == torch::kInt32 || dtype == torch::kInt64) {
        return torch::randint(max_val, {dim0_size, dim1_size}, dtype);
    }
    return torch::randn({dim0_size, dim1_size}, dtype);
}

int genRandTensorAndWriteToFile(torch::Tensor &rand_tensor, int total_embeddings, int embedding_size, torch::Dtype dtype, int fd) {
    rand_tensor = getRandTensor(total_embeddings, embedding_size, dtype);
    int tensor_size = embedding_size * get_dtype_size_wrapper(dtype);
    return pwrite_wrapper(fd, rand_tensor.data_ptr(), total_embeddings * tensor_size, 0);
}

bool checkPermOf2dTensor(torch::Tensor &a, torch::Tensor &b) {
    if (a.sizes().size() != b.sizes().size() || a.sizes().size() != 2) return false;
    vector<int> has_seen_count(a.size(0), 0);
    for (int i = 0; i < a.size(0); i++) {
        for (int j = 0; j < b.size(0); j++) {
            if (a[i].equal(b[j])) {
                has_seen_count[i] += 1;
            }
        }
    }
    for (int i = 0; i < a.size(0); i++)
        if (has_seen_count[i] != 1) return false;
    return true;
}

void sortWithinEdgeBuckets(torch::Tensor &rand_tensor, vector<int64_t> &edge_bucket_sizes, int sort_dim) {
    int64_t offset = 0;
    for (auto itr = edge_bucket_sizes.begin(); itr != edge_bucket_sizes.end(); itr++) {
        torch::Tensor edge_bucket = rand_tensor.slice(0, offset, offset + *itr);
        edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
        rand_tensor.slice(0, offset, offset + *itr) = edge_bucket;
        edge_bucket = torch::Tensor();
        offset += *itr;
    }
}

bool sortEdgesSrcDest(vector<int> &edge1, vector<int> &edge2) {
    if (edge1[0] != edge2[0]) return edge1[0] < edge2[0];
    if (edge1[2] != edge2[2]) return edge1[2] < edge2[2];
    return false;
}

vector<int64_t> partitionEdges(torch::Tensor &edges, int num_partitions, int num_nodes) {
    vector<vector<int>> edges_vec(edges.size(0), vector<int>(edges.size(1)));
    for (int i = 0; i < edges_vec.size(); i++) {
        for (int j = 0; j < edges_vec[i].size(); j++) {
            edges_vec[i][j] = edges[i][j].item<int>();
        }
    }
    sort(edges_vec.begin(), edges_vec.end(), sortEdgesSrcDest);
    int partition_size = ceil(((double)num_nodes) / num_partitions);
    std::pair<int, int> prev(edges_vec[0][0] / partition_size, edges_vec[0][2] / partition_size), cur;
    int count = 1;
    vector<int64_t> partition_sizes_;
    for (int i = 1; i < edges_vec.size(); i++) {
        cur = std::pair<int, int>(edges_vec[i][0] / partition_size, edges_vec[i][2] / partition_size);
        if (cur == prev) {
            count++;
            continue;
        }
        partition_sizes_.push_back(count);
        count = 1;
        prev = cur;
    }
    partition_sizes_.push_back(count);
    return partition_sizes_;
}