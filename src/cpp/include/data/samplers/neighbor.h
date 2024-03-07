//
// Created by Jason Mohoney on 2/8/22.
//

#ifndef MARIUS_NEIGHBOR_SAMPLER_H
#define MARIUS_NEIGHBOR_SAMPLER_H

#include "configuration/config.h"
#include "storage/graph_storage.h"
#include "data/features_loader.h"

std::tuple<torch::Tensor, torch::Tensor> sample_all_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                        torch::Tensor num_neighbors);

std::tuple<torch::Tensor, torch::Tensor> sample_all_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                        torch::Tensor num_neighbors, int64_t total_neighbors);

std::tuple<torch::Tensor, torch::Tensor> sample_uniform_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, int64_t max_neighbors, int64_t max_id);

std::tuple<torch::Tensor, torch::Tensor> sample_uniform_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, int64_t max_neighbors, int64_t total_neighbors);

std::tuple<torch::Tensor, torch::Tensor> sample_dropout_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, float rate);

std::tuple<torch::Tensor, torch::Tensor> sample_dropout_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, float rate, int64_t total_neighbors);

/**
 * Samples the neighbors from a given batch given a neighbor sampling strategy.
 */
class NeighborSampler {
   public:
    shared_ptr<GraphModelStorage> storage_;
    shared_ptr<MariusGraph> graph_;

    virtual ~NeighborSampler(){};

    /**
     * Get neighbors of provided nodes using given neighborhood sampling strategy.
     * @param node_ids Nodes to get neighbors from
     * @return The neighbors sampled using strategy
     */
    virtual DENSEGraph getNeighbors(torch::Tensor node_ids, shared_ptr<MariusGraph> graph = nullptr, int worker_id = 0) = 0;

    virtual int64_t getNeighborsPages(torch::Tensor node_ids, shared_ptr<MariusGraph> graph = nullptr, int worker_id = 0) = 0;
};

class LayeredNeighborSampler : public NeighborSampler {
   public:
    bool use_incoming_nbrs_;
    bool use_outgoing_nbrs_;
    std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers_;
    std::shared_ptr<FeaturesLoader> features_loader_;
    torch::Tensor in_mem_nodes_;
    float percent_removed_total_ = 0.0;
    int64_t percent_count_ = 0;
    float scaling_factor_total_ = 0.0;
    int64_t scaling_count_ = 0;

    bool use_hashmap_sets_;
    bool use_bitmaps_;

    // TODO: this change may affect test, docs, python examples
    LayeredNeighborSampler(shared_ptr<GraphModelStorage> storage, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, bool use_incoming_nbrs = true,
                           bool use_outgoing_nbrs = true);

    LayeredNeighborSampler(shared_ptr<MariusGraph> graph, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, bool use_incoming_nbrs = true,
                           bool use_outgoing_nbrs = true);
    
    LayeredNeighborSampler(shared_ptr<MariusGraph> graph, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, torch::Tensor in_mem_nodes, 
                           shared_ptr<FeaturesLoaderConfig> features_config, bool use_incoming_nbrs = false, bool use_outgoing_nbrs = true);

    LayeredNeighborSampler(std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, bool use_incoming_nbrs = true, bool use_outgoing_nbrs = true);

    void checkLayerConfigs();

    torch::Tensor remove_in_mem_nodes(torch::Tensor node_ids);

    DENSEGraph getNeighbors(torch::Tensor node_ids, shared_ptr<MariusGraph> graph = nullptr, int worker_id = 0) override;

    float getAvgScalingFactor();

    float getAvgPercentRemoved();

    int64_t getNeighborsPages(torch::Tensor node_ids, shared_ptr<MariusGraph> graph = nullptr, int worker_id = 0) override;

    torch::Tensor computeDeltaIdsHelperMethod1(torch::Tensor hash_map, torch::Tensor node_ids, torch::Tensor delta_incoming_edges,
                                               torch::Tensor delta_outgoing_edges, int64_t num_nodes_in_memory);
};

#endif  // MARIUS_NEIGHBOR_SAMPLER_H