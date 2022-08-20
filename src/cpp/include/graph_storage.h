//
// Created by Jason Mohoney on 6/18/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
#define MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_

#include "configuration/constants.h"
#include "model.h"
#include "storage.h"

struct GraphModelStoragePtrs {
    Storage *edges;
    Storage *edge_features;
    Storage *train_edges;
    Storage *train_edges_dst_sort;
    Storage *train_edges_features;
    Storage *validation_edges;
    Storage *validation_edges_features;
    Storage *test_edges;
    Storage *test_edges_features;
    Storage *nodes;
    Storage *train_nodes;
    Storage *valid_nodes;
    Storage *test_nodes;
    Storage *node_features;
    Storage *node_labels;
    Storage *relation_features;
    Storage *relation_labels;
    Storage *node_embeddings;
    Storage *node_optimizer_state;
};

struct InMemorySubgraphState {
    EdgeList all_in_memory_edges_;
    EdgeList all_in_memory_edges_dst_sort_;
    EdgeList all_in_memory_mapped_edges_;
    torch::Tensor in_memory_partition_ids_;
    torch::Tensor in_memory_edge_bucket_ids_;
    torch::Tensor in_memory_edge_bucket_sizes_;
    torch::Tensor in_memory_edge_bucket_starts_;
    torch::Tensor in_memory_edge_bucket_ids_dst_;
    torch::Tensor in_memory_edge_bucket_sizes_dst_;
    torch::Tensor in_memory_edge_bucket_starts_dst_;
    torch::Tensor global_to_local_index_map_;
    MariusGraph *in_memory_subgraph_;
};

class GraphModelStorage {
  private:
    void _load(Storage *storage);

    void _unload(Storage *storage, bool write);

  protected:
    bool train_;

    int64_t num_edges_;
    int64_t num_nodes_;

    // used for evaluation
    map<std::pair<int, int>, vector<int>> src_map_;           /**< Map keyed by the source node and relation ids, where the destination node is the value. Provides fast lookups for edge existence */
    map<std::pair<int, int>, vector<int>> dst_map_;           /**< Map keyed by the destination node and relation ids, where the source node is the value. Provides fast lookups for edge existence */
    InMemory *in_memory_embeddings_;
    InMemory *in_memory_features_;

  public:
    // In memory subgraph for partition buffer

    EdgeList active_edges_;
    Indices active_nodes_;

    std::mutex *subgraph_lock_;
    std::condition_variable *subgraph_cv_;
    InMemorySubgraphState *current_subgraph_state_;
    InMemorySubgraphState *next_subgraph_state_;
    bool prefetch_;
    bool prefetch_complete_;

    GraphModelStoragePtrs storage_ptrs_;
    shared_ptr<StorageConfig> storage_config_;
    LearningTask learning_task_;
    bool full_graph_evaluation_;
    bool filtered_eval_;

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs,
                      shared_ptr<StorageConfig> storage_config,
                      LearningTask learning_task=LearningTask::LINK_PREDICTION,
                      bool filtered_eval=false);

    ~GraphModelStorage();

    void load();

    void unload(bool write);

    void initializeInMemorySubGraph(torch::Tensor buffer_state);

    void updateInMemorySubGraph_(InMemorySubgraphState *subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids);

    void updateInMemorySubGraph();

    void getNextSubGraph();

    EdgeList merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src);

    void setEvalFilter(Batch *batch);

    void setEdgesStorage(Storage *edge_storage);

    void setNodesStorage(Storage *node_storage);

    EdgeList getEdges(Indices indices);

    EdgeList getEdgesRange(int64_t start, int64_t size);

    Indices getRandomNodeIds(int64_t size);

    Indices getNodeIdsRange(int64_t start, int64_t size);

    void shuffleEdges();

    Embeddings getNodeEmbeddings(Indices indices);

    Embeddings getNodeEmbeddingsRange(int64_t start, int64_t size);

    Features getNodeFeatures(Indices indices);

    Features getNodeFeaturesRange(int64_t start, int64_t size);

    Labels getNodeLabels(Indices indices);

    Labels getNodeLabelsRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddings(Indices indices, Embeddings embeddings);

    void updateAddNodeEmbeddings(Indices indices, torch::Tensor values);

    OptimizerState getNodeEmbeddingState(Indices indices);

    OptimizerState getNodeEmbeddingStateRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddingState(Indices indices, OptimizerState state);

    void updateAddNodeEmbeddingState(Indices indices, torch::Tensor values);

    bool embeddingsOffDevice();

    int getNumPartitions() {
        int num_partitions = 1;

        if (storage_config_->features != nullptr) {
            num_partitions = std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config_->features->options)->num_partitions;
        }

        if (storage_config_->embeddings != nullptr) {
            num_partitions = std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config_->embeddings->options)->num_partitions;
        }

        return num_partitions;
    }

    bool useInMemorySubGraph() {
        bool embeddings_buffered = false;
        bool features_buffered = false;

        if (storage_config_->embeddings != nullptr) {
            StorageBackend embeddings_backend = storage_config_->embeddings->type;
            embeddings_buffered = (embeddings_backend == StorageBackend::PARTITION_BUFFER);
        }

        if (storage_config_->features != nullptr) {
            StorageBackend features_backend = storage_config_->features->type;
            features_buffered = (features_backend == StorageBackend::PARTITION_BUFFER);
        }

        return (embeddings_buffered || features_buffered) && (train_ || (!full_graph_evaluation_));
    }

    bool hasSwap() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->hasSwap();
        }

        if (storage_ptrs_.node_features != nullptr) {
            return ((PartitionBufferStorage *) storage_ptrs_.node_features)->hasSwap();
        }

        return false;
    }

    std::pair<std::vector<int>, std::vector<int>> getNextSwapIds() {
        std::vector<int> evict_ids;
        std::vector<int> admit_ids;

        if (storage_ptrs_.node_embeddings != nullptr && storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            evict_ids = ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getNextEvict();
            admit_ids = ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getNextAdmit();
        } else if (storage_ptrs_.node_features != nullptr && storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
            evict_ids = ((PartitionBufferStorage *) storage_ptrs_.node_features)->getNextEvict();
            admit_ids = ((PartitionBufferStorage *) storage_ptrs_.node_features)->getNextAdmit();
        }

        return std::make_pair(evict_ids, admit_ids);
    }

    void performSwap() {
        if (storage_ptrs_.node_embeddings != nullptr && storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->performNextSwap();
            ((PartitionBufferStorage *) storage_ptrs_.node_optimizer_state)->performNextSwap();
        }

        if (storage_ptrs_.node_features != nullptr && storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
            ((PartitionBufferStorage *) storage_ptrs_.node_features)->performNextSwap();
        }
    }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) {
        if (storage_ptrs_.node_embeddings != nullptr && storage_config_->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            ((PartitionBufferStorage *) storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
        }
        if (storage_ptrs_.node_features != nullptr && storage_config_->features->type == StorageBackend::PARTITION_BUFFER) {
            ((PartitionBufferStorage *) storage_ptrs_.node_features)->setBufferOrdering(buffer_states);
        }
    }

    void setActiveEdges(torch::Tensor active_edges) {
        active_edges_ = active_edges;
    }

    void setActiveNodes(torch::Tensor node_ids) {
        active_nodes_ = node_ids;
    }


    int64_t getNumActiveEdges() {
        if (active_edges_.defined()) {
            return active_edges_.size(0);
        } else {
            return storage_ptrs_.edges->getDim0();
        }
    }

    int64_t getNumActiveNodes() {
        if (active_nodes_.defined()) {
            return active_nodes_.size(0);
        } else {
            return storage_ptrs_.nodes->getDim0();
        }
    }

    int64_t getNumEdges() {
        return num_edges_;
    }

    int64_t getNumNodes() {
        return num_nodes_;
    }

    int64_t getNumNodesInMemory() {

        if (storage_config_->embeddings != nullptr) {
            if (useInMemorySubGraph()) {
                return ((PartitionBufferStorage *) storage_ptrs_.node_embeddings)->getNumInMemory();
            }
        }

        if (storage_config_->features != nullptr) {
            if (useInMemorySubGraph()) {
                return ((PartitionBufferStorage *) storage_ptrs_.node_features)->getNumInMemory();
            }
        }

        return storage_config_->dataset->num_nodes;
    }

    void setTrainSet() {
        train_ = true;
        unload(false);
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setEdgesStorage(storage_ptrs_.train_edges);
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            setNodesStorage(storage_ptrs_.train_nodes);
        }
    }

    void setValidationSet() {
        train_ = false;
        unload(false);

        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setEdgesStorage(storage_ptrs_.validation_edges);
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            setNodesStorage(storage_ptrs_.valid_nodes);
        }
    }

    void setTestSet() {
        train_ = false;
        unload(false);

        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setEdgesStorage(storage_ptrs_.test_edges);
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            setNodesStorage(storage_ptrs_.test_nodes);
        }
    }
};


#endif //MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
