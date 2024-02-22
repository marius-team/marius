//
// Created by Jason Mohoney on 8/25/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_GRAPH_H_
#define MARIUS_SRC_CPP_INCLUDE_GRAPH_H_

#include "common/datatypes.h"
#include "common/util.h"
#include "configuration/config.h"
#include "nn/layers/gnn/layer_helpers.h"

/**
 * Object to handle arbitrary in-memory graph/sub-graph.
 */
class MariusGraph {
   public:
    EdgeList src_sorted_edges_;           // easy access of outgoing neighbors
    EdgeList dst_sorted_edges_;           // easy access of incoming neighbors
    EdgeList active_in_memory_subgraph_;  // shuffled

    int64_t num_nodes_in_memory_;
    Indices node_ids_;
    Indices out_sorted_uniques_;
    Indices out_offsets_;
    torch::Tensor out_num_neighbors_;
    Indices in_sorted_uniques_;
    Indices in_offsets_;
    torch::Tensor in_num_neighbors_;

    int max_out_num_neighbors_;
    int max_in_num_neighbors_;

    int num_hash_maps_;
    std::vector<torch::Tensor> hash_maps_;

    // used for filtering negatives
    EdgeList all_src_sorted_edges_;
    EdgeList all_dst_sorted_edges_;

    MariusGraph();

    MariusGraph(EdgeList edges);

    MariusGraph(EdgeList src_sorted_edges, EdgeList dst_sorted_edges, int64_t num_nodes_in_memory, int num_hash_maps = 1);
    // TODO: this change may affect some cpp and python tests

    ~MariusGraph();

    /**
     * Get the node IDs from the graph.
     * @return Tensor of node IDs
     */
    Indices getNodeIDs();

    /**
     * Get the edges from the graph.
     * @param incoming Get incoming edges if true, outgoing edges if false
     * @return Tensor of edge IDs
     */
    Indices getEdges(bool incoming = true);

    /**
     * Get the relation IDs from the graph.
     * @param incoming Get incoming relation IDs if true, outgoing relation IDs if false
     * @return Tensor of relation IDs
     */
    Indices getRelationIDs(bool incoming = true);

    /**
     * Get the offsets of the neighbors in the sorted edge list.
     * @param incoming Get incoming neighbor offsets if true, outgoing neighbor offsets if false
     * @return Tensor of neighbor offsets
     */
    Indices getNeighborOffsets(bool incoming = true);

    /**
     * Get the number of neighbors for each node in the graph.
     * @param incoming Get number of incoming neighbor if true, number of outgoing neighbors if false
     * @return Number of neighbors
     */
    torch::Tensor getNumNeighbors(bool incoming = true);

    /**
     * Get the neighbors for the specified node IDs.
     * @param node_ids The node IDs to get neighbors from
     * @param incoming Get incoming neighbors if true, outgoing if false
     * @param neighbor_sampling_layer The neighbor sampling strategy to use
     * @param max_neighbors_size The maximum number of neighbors to sample
     * @return Neighbors of specified nodes
     */
    std::tuple<torch::Tensor, torch::Tensor> getNeighborsForNodeIds(torch::Tensor node_ids, bool incoming, NeighborSamplingLayer neighbor_sampling_layer,
                                                                    int max_neighbors_size, float rate);

    /**
     * Clear the graph.
     */
    void clear();

    void to(torch::Device device);

    void sortAllEdges(EdgeList additional_edges);
};

/**
 * MariusGraph sublass, orders the CSR representation of the graph for fast GNN encoding.
 */
class DENSEGraph : public MariusGraph {
   public:
    Indices hop_offsets_;

    Indices in_neighbors_mapping_;
    Indices out_neighbors_mapping_;

    std::vector<torch::Tensor> in_neighbors_vec_;
    std::vector<torch::Tensor> out_neighbors_vec_;

    torch::Tensor node_properties_;

    int num_nodes_in_memory_;

    DENSEGraph();

    DENSEGraph(Indices hop_offsets, Indices node_ids, Indices in_offsets, std::vector<torch::Tensor> in_neighbors_vec, Indices in_neighbors_mapping,
               Indices out_offsets, std::vector<torch::Tensor> out_neighbors_vec, Indices out_neighbors_mapping, int num_nodes_in_memory);

    ~DENSEGraph();

    /**
     * Prepares GNN graph for next layer.
     */
    void prepareForNextLayer();

    /**
     * Gets the ids of the neighbors for the current layer.
     * @param incoming Get incoming edges if true, outgoing edges if false
     * @param global If false, return node IDs local to the batch. If true, return any global node IDs
     * @return Tensor of edge IDs
     */
    Indices getNeighborIDs(bool incoming = true, bool global = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getCombinedNeighborIDs();

    /**
     * Gets the offset of the node ids in the outermost layer.
     * @return Layer offset
     */
    int64_t getLayerOffset();

    /**
     * Maps local IDs to batch.
     */
    void performMap();

    void setNodeProperties(torch::Tensor node_properties);

    /**
     * Clear the graph.
     */
    void clear();

    void to(torch::Device device, CudaStream *compute_stream = nullptr, CudaStream *transfer_stream = nullptr);
};

#endif  // MARIUS_SRC_CPP_INCLUDE_GRAPH_H_
