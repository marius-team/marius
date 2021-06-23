//
// Created by Jason Mohoney on 6/18/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
#define MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_

#include "config.h"
#include "storage.h"

struct GraphModelStoragePtrs {
    Storage *edges;
    Storage *edge_features;
    Storage *node_embedding_storage;
    Storage *node_optimizer_state;
    Storage *node_feature_storage;
    Storage *lhs_relations_storage;
    Storage *lhs_relations_optimizer_state;
    Storage *rhs_relations_storage;
    Storage *rhs_relations_optimizer_state;
    Storage *rhs_relations_feature_storage;
    Storage *relations_storage;
    Storage *relations_optimizer_state;
    Storage *relation_feature_storage;
};

class GraphModelStorage {
  private:
    void initializeInMemorySubGraph(std::vector<int> buffer_state) ;

    void updateInMemorySubGraph(int admit_partition_id, int evict_partition_id);

  protected:
    bool featurized_edges_;
    bool featurized_nodes_;
    bool featurized_relations_;
    bool stateful_optimizer_nodes_;
    bool stateful_optimizer_relations_;
    bool partitioned_nodes_;

    bool train_;
    DataSetType data_set_type_;

    GraphModelStoragePtrs storage_ptrs_;

    // In memory subgraph for partition buffer
    torch::Tensor in_memory_partition_ids_;
    torch::Tensor in_memory_edge_bucket_ids_;
    torch::Tensor in_memory_edge_bucket_starts_;
    torch::Tensor in_memory_edge_bucket_sizes_;

    EdgeList in_memory_subgraph_;
    torch::Tensor src_sorted_list_;
    torch::Tensor dst_sorted_list_;

  public:
    GraphModelStorage(const MariusOptions &options, DataSetType data_set_type, bool train);

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, const StorageOptions &storage_options, DataSetType data_set_type, bool train);

    ~GraphModelStorage();

    void load();

    void unload(bool write);

    void setEdgesStorage(Storage *edge_storage);

    EdgeList getEdges(Indices indices);

    EdgeList getEdgesRange(int64_t start, int64_t size);

    void updateEdges(Indices indices, EdgeList edges);

    void shuffleEdges();

    Embeddings getNodeEmbeddings(Indices indices);

    Embeddings getNodeEmbeddingsRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddings(Indices indices, Embeddings embeddings);

    void updateAddNodeEmbeddings(Indices indices, torch::Tensor values);

    OptimizerState getNodeEmbeddingState(Indices indices);

    OptimizerState getNodeEmbeddingStateRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddingState(Indices indices, OptimizerState state);

    void updateAddNodeEmbeddingState(Indices indices, torch::Tensor values);

    Relations getRelations(Indices indices, bool lhs = true);

    Relations getRelationsRange(int64_t start, int64_t size, bool lhs = true);

    void updatePutRelations(Indices indices, Relations embeddings, bool lhs = true);

    void updateAddRelations(Indices indices, torch::Tensor values, bool lhs = true);

    OptimizerState getRelationsState(Indices indices, bool lhs = true);

    OptimizerState getRelationsStateRange(int64_t start, int64_t size, bool lhs = true);

    void updatePutRelationsState(Indices indices, OptimizerState state, bool lhs = true);

    void updateAddRelationsState(Indices indices,torch::Tensor values, bool lhs = true);

    std::tuple<torch::Tensor, torch::Tensor> gatherNeighbors(torch::Tensor node_ids, bool src);
};


#endif //MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
