//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_GRAPH_SAMPLERS_H_
#define MARIUS_SRC_CPP_INCLUDE_GRAPH_SAMPLERS_H_

#include "graph_storage.h"

#include "configuration/config.h"

/**
 * Samples the edges from a given batch.
 */
class EdgeSampler {
  public:
    GraphModelStorage *graph_storage_;

    virtual ~EdgeSampler() { };

    /**
     * Get edges for a given batch.
     * @param batch Batch to sample into
     * @return Edges sampled for the batch
     */
    virtual EdgeList getEdges(Batch *batch) = 0;
};

class RandomEdgeSampler : public EdgeSampler {
  public:
    bool without_replacement_;

    RandomEdgeSampler(GraphModelStorage *graph_storage, bool without_replacement = true);

    EdgeList getEdges(Batch *batch) override;
};

/**
 * Samples the negative edges from a given batch.
 */
class NegativeSampler {
  public:
    GraphModelStorage *graph_storage_;

    std::mutex sampler_lock_;

    virtual ~NegativeSampler() { };

    /**
     * Get negative edges from the given batch.
     * Return a tensor of node ids of shape [num_negs] or a [num_negs, 3] shaped tensor of negative edges.
     * @param batch Batch to sample into
     * @param src Source
     * @return The negative nodes/edges sampled
     */
    virtual torch::Tensor getNegatives(Batch *batch, bool src) = 0;

    void lock();

    void unlock();
};

class RandomNegativeSampler : public NegativeSampler {
  public:
    bool without_replacement_;
    int num_chunks_;
    int num_negatives_;

    RandomNegativeSampler(GraphModelStorage *graph_storage,
                          int num_chunks,
                          int num_negatives,
                          bool without_replacement = true);

    torch::Tensor getNegatives(Batch *batch, bool src = true) override;
};

class FilteredNegativeSampler : public NegativeSampler {
  public:
    FilteredNegativeSampler(GraphModelStorage *graph_storage);

    torch::Tensor getNegatives(Batch *batch, bool src = true) override;
};

/**
 * Samples the neighbors from a given batch given a neighbor sampling strategy.
 */
class NeighborSampler {
  public:
    GraphModelStorage *storage_;

    bool incoming_;
    bool outgoing_;

    virtual ~NeighborSampler() { };

    /**
     * Get neighbors of provided nodes using given neighborhood sampling strategy.
     * @param node_ids Nodes to get neighbors from
     * @return The neighbors sampled using strategy
     */
    virtual GNNGraph getNeighbors(torch::Tensor node_ids) = 0;
};

class LayeredNeighborSampler : public NeighborSampler {
public:
    bool use_hashmap_sets_;

    std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers_;

    LayeredNeighborSampler(GraphModelStorage *storage,
                           std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs,
                           bool incoming,
                           bool outgoing,
                           bool use_hashmap_sets);

    GNNGraph getNeighbors(torch::Tensor node_ids) override;

    torch::Tensor computeDeltaIdsHelperMethod1(torch::Tensor hash_map, torch::Tensor node_ids,
                                               torch::Tensor delta_incoming_edges, torch::Tensor delta_outgoing_edges,
                                               int64_t num_nodes_in_memory);
};



#endif //MARIUS_SRC_CPP_INCLUDE_GRAPH_SAMPLERS_H_
