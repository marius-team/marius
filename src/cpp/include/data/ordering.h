//
// Created by Jason Mohoney on 7/17/20.
//

#ifndef MARIUS_ORDERING_H
#define MARIUS_ORDERING_H

#include "batch.h"

using std::pair;

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets, shared_ptr<c10d::ProcessGroupGloo> pg_gloo,
                                                                               shared_ptr<DistributedConfig> dist_config);

vector<torch::Tensor> convertToVectorOfTensors(vector<vector<int>> buffer_states,
                                               torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

vector<vector<int>> convertToVectorOfInts(vector<torch::Tensor> buffer_states);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer);

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity);

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

vector<vector<int>> convertToFinePartitions(vector<vector<int>> coarse_buffer_states, int num_partitions, int buffer_capacity,
                                            int fine_to_coarse_ratio, int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering();

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions,
                                                                                  shared_ptr<c10d::ProcessGroupGloo> pg_gloo, shared_ptr<DistributedConfig> dist_config);

vector<torch::Tensor> getDispersedOrderingHelper(int num_partitions, int buffer_capacity);

vector<torch::Tensor> randomlyAssignTrainNodesToBuffers(vector<torch::Tensor> buffer_states, Indices train_nodes,
                                                        int64_t total_num_nodes, int num_partitions, int buffer_capacity);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
                                                                                           int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDiagOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio, int num_cache_partitions,
                                                                         bool randomly_assign_edge_buckets = true, Indices train_nodes = torch::Tensor(), int64_t total_num_nodes = -1,
                                                                         shared_ptr<c10d::ProcessGroupGloo> pg_gloo = nullptr,
                                                                         shared_ptr<DistributedConfig> dist_config = nullptr,
                                                                         bool sequential = false, bool in_memory = false);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering();

#endif  // MARIUS_ORDERING_H
