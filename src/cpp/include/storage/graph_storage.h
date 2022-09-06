//
// Created by Jason Mohoney on 6/18/21.
//

#ifndef MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
#define MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_

#include "configuration/constants.h"
#include "nn/model.h"
#include "storage/storage.h"

struct GraphModelStoragePtrs {
    shared_ptr<Storage> edges = nullptr;
    shared_ptr<Storage> train_edges = nullptr;
    shared_ptr<Storage> validation_edges = nullptr;
    shared_ptr<Storage> test_edges = nullptr;
    shared_ptr<Storage> nodes = nullptr;
    shared_ptr<Storage> train_nodes = nullptr;
    shared_ptr<Storage> valid_nodes = nullptr;
    shared_ptr<Storage> test_nodes = nullptr;
    shared_ptr<Storage> node_features = nullptr;
    shared_ptr<Storage> node_labels = nullptr;
    shared_ptr<Storage> relation_features = nullptr;
    shared_ptr<Storage> relation_labels = nullptr;
    shared_ptr<Storage> node_embeddings = nullptr;
    shared_ptr<Storage> encoded_nodes = nullptr;
    shared_ptr<Storage> node_optimizer_state = nullptr;
    std::vector<shared_ptr<Storage>> filter_edges;
};

struct InMemorySubgraphState {
    EdgeList all_in_memory_edges_;
    EdgeList all_in_memory_mapped_edges_;
    torch::Tensor in_memory_partition_ids_;
    torch::Tensor in_memory_edge_bucket_ids_;
    torch::Tensor in_memory_edge_bucket_sizes_;
    torch::Tensor in_memory_edge_bucket_starts_;
    torch::Tensor global_to_local_index_map_;
    shared_ptr<MariusGraph> in_memory_subgraph_;
};

class GraphModelStorage {
   private:
    void _load(shared_ptr<Storage> storage);

    void _unload(shared_ptr<Storage> storage, bool write);

    int64_t num_nodes_;
    int64_t num_edges_;

   protected:
    bool train_;

    shared_ptr<InMemory> in_memory_embeddings_;
    shared_ptr<InMemory> in_memory_features_;

   public:
    // In memory subgraph for partition buffer

    EdgeList active_edges_;
    Indices active_nodes_;

    std::mutex *subgraph_lock_;
    std::condition_variable *subgraph_cv_;
    shared_ptr<InMemorySubgraphState> current_subgraph_state_;
    shared_ptr<InMemorySubgraphState> next_subgraph_state_;
    bool prefetch_;
    bool prefetch_complete_;

    GraphModelStoragePtrs storage_ptrs_;
    bool full_graph_evaluation_;

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config);

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch = false);

    ~GraphModelStorage();

    void load();

    void unload(bool write);

    void initializeInMemorySubGraph(torch::Tensor buffer_state);

    void updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids);

    void updateInMemorySubGraph();

    void getNextSubGraph();

    EdgeList merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src);

    void setEdgesStorage(shared_ptr<Storage> edge_storage);

    void setNodesStorage(shared_ptr<Storage> node_storage);

    EdgeList getEdges(Indices indices);

    EdgeList getEdgesRange(int64_t start, int64_t size);

    Indices getRandomNodeIds(int64_t size);

    Indices getNodeIdsRange(int64_t start, int64_t size);

    void shuffleEdges();

    torch::Tensor getNodeEmbeddings(Indices indices);

    torch::Tensor getNodeEmbeddingsRange(int64_t start, int64_t size);

    torch::Tensor getNodeFeatures(Indices indices);

    torch::Tensor getNodeFeaturesRange(int64_t start, int64_t size);

    torch::Tensor getEncodedNodes(Indices indices);

    torch::Tensor getEncodedNodesRange(int64_t start, int64_t size);

    torch::Tensor getNodeLabels(Indices indices);

    torch::Tensor getNodeLabelsRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddings(Indices indices, torch::Tensor values);

    void updateAddNodeEmbeddings(Indices indices, torch::Tensor values);

    void updatePutEncodedNodes(Indices indices, torch::Tensor values);

    void updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values);

    OptimizerState getNodeEmbeddingState(Indices indices);

    OptimizerState getNodeEmbeddingStateRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddingState(Indices indices, OptimizerState state);

    void updateAddNodeEmbeddingState(Indices indices, torch::Tensor values);

    bool embeddingsOffDevice();

    void sortAllEdges();

    int getNumPartitions() {
        int num_partitions = 1;

        if (useInMemorySubGraph()) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->options_->num_partitions;
            }

            // assumes both the node features and node embeddings have the same number of partitions
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->options_->num_partitions;
            }
        }

        return num_partitions;
    }

    bool useInMemorySubGraph() {
        bool embeddings_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings);
        bool features_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features);

        return (embeddings_buffered || features_buffered) && (train_ || (!full_graph_evaluation_));
    }

    bool hasSwap() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->hasSwap();
        }

        if (storage_ptrs_.node_features != nullptr) {
            return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->hasSwap();
        }

        return false;
    }

    std::pair<std::vector<int>, std::vector<int>> getNextSwapIds() {
        std::vector<int> evict_ids;
        std::vector<int> admit_ids;

        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextAdmit();
        } else if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextAdmit();
        }

        return std::make_pair(evict_ids, admit_ids);
    }

    void performSwap() {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->performNextSwap();
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->performNextSwap();
            }
        }

        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->performNextSwap();
        }
    }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state != nullptr && !train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->setBufferOrdering(buffer_states);
        }
    }

    void setActiveEdges(torch::Tensor active_edges) { active_edges_ = active_edges; }

    void setActiveNodes(torch::Tensor node_ids) { active_nodes_ = node_ids; }

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

    int64_t getNumEdges() { return storage_ptrs_.edges->getDim0(); }

    int64_t getNumNodes() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->getDim0();
        }

        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->getDim0();
        }

        return num_nodes_;
    }

    int64_t getNumNodesInMemory() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (useInMemorySubGraph()) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNumInMemory();
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (useInMemorySubGraph()) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNumInMemory();
            }
        }

        return getNumNodes();
    }

    void setTrainSet() {
        train_ = true;

        if (storage_ptrs_.train_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.train_edges);
        }

        if (storage_ptrs_.train_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.train_nodes);
        }
    }

    void setValidationSet() {
        train_ = false;

        if (storage_ptrs_.validation_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.validation_edges);
        }

        if (storage_ptrs_.valid_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.valid_nodes);
        }
    }

    void setTestSet() {
        train_ = false;

        if (storage_ptrs_.test_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.test_edges);
        }

        if (storage_ptrs_.test_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.test_nodes);
        }
    }

    void setFilterEdges(std::vector<shared_ptr<Storage>> filter_edges) { storage_ptrs_.filter_edges = filter_edges; }

    void addFilterEdges(shared_ptr<Storage> filter_edges) { storage_ptrs_.filter_edges.emplace_back(filter_edges); }
};

#endif  // MARIUS_SRC_CPP_INCLUDE_GRAPH_STORAGE_H_
