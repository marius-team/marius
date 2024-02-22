#include <fcntl.h>
#include <torch/torch.h>
#include <unistd.h>

#include <string>

#include "storage/storage.h"

int createTmpFile(std::string &filename);

torch::Tensor getRandTensor(int dim0_size, int dim1_size, torch::Dtype dtype, int max_val = 1000);

int genRandTensorAndWriteToFile(torch::Tensor &rand_tensor, int total_embeddings, int embedding_size, torch::Dtype dtype, int fd);

bool checkPermOf2dTensor(torch::Tensor &a, torch::Tensor &b);

void sortWithinEdgeBuckets(torch::Tensor &rand_tensor, vector<int64_t> &edge_bucket_sizes, int sort_dim = 0);

vector<int64_t> partitionEdges(torch::Tensor &edges, int num_partitions, int num_nodes);