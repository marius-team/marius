//
// Created by Jason Mohoney on 7/17/20.
//

#ifndef MARIUS_ORDERING_H
#define MARIUS_ORDERING_H

#include "batch.h"

using std::pair;

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer);

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity);

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering();

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
                                                                                           int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering();

#endif  // MARIUS_ORDERING_H
