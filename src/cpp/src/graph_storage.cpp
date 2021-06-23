//
// Created by Jason Mohoney on 6/18/21.
//

#include "graph_storage.h"

#include "io.h"


GraphModelStorage::GraphModelStorage(const MariusOptions &marius_options, DataSetType data_set_type, bool train) {
    featurized_edges_ = false;
    featurized_nodes_ = false;
    featurized_relations_ = false;
    stateful_optimizer_nodes_ = false;
    stateful_optimizer_relations_ = false;
    partitioned_nodes_ = false;

    train_ = train;
    data_set_type_ = data_set_type;

    GraphModelStoragePtrs storage_ptrs = {};

    if (marius_options.storage.num_partitions > 1) {
        partitioned_nodes_ = true;
    }

    if (train) {
        if (marius_options.training.optimizer_type == OptimizerType::Adagrad) {
            stateful_optimizer_nodes_ = true;
        }

        if (marius_options.training.optimizer_type == OptimizerType::Adagrad) {
            stateful_optimizer_relations_ = true;
        }

        tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> train_storage_ptrs = initializeTrain();

        Storage *train_edges = std::get<0>(train_storage_ptrs);
        Storage *node_embeddings = std::get<3>(train_storage_ptrs);
        Storage *node_emb_state = std::get<4>(train_storage_ptrs);
        Storage *lhs_rel = std::get<5>(train_storage_ptrs);
        Storage *lhs_rel_state = std::get<6>(train_storage_ptrs);
        Storage *rhs_rel = std::get<7>(train_storage_ptrs);
        Storage *rhs_rel_state = std::get<8>(train_storage_ptrs);

        storage_ptrs.edges = train_edges;
        storage_ptrs.node_embedding_storage = node_embeddings;
        storage_ptrs.node_optimizer_state = node_emb_state;
        storage_ptrs.lhs_relations_storage = lhs_rel;
        storage_ptrs.lhs_relations_optimizer_state = lhs_rel_state;
        storage_ptrs.rhs_relations_storage = rhs_rel;
        storage_ptrs.rhs_relations_optimizer_state = rhs_rel_state;
    } else {
        // train, validation or test set
        tuple<Storage *, Storage *, Storage *, Storage *> eval_storage_ptrs = initializeEval(data_set_type_);

        Storage *eval_edges = std::get<0>(eval_storage_ptrs);
        Storage *node_embeddings = std::get<1>(eval_storage_ptrs);
        Storage *lhs_rel = std::get<3>(eval_storage_ptrs);
        Storage *rhs_rel = std::get<3>(eval_storage_ptrs);

        storage_ptrs.edges = eval_edges;
        storage_ptrs.node_embedding_storage = node_embeddings;
        storage_ptrs.lhs_relations_storage = lhs_rel;
        storage_ptrs.rhs_relations_storage = rhs_rel;
    }

    storage_ptrs_ = storage_ptrs;
}

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, const StorageOptions &storage_options, DataSetType data_set_type, bool train) {
    featurized_edges_ = false;
    featurized_nodes_ = false;
    featurized_relations_ = false;
    stateful_optimizer_nodes_ = false;
    stateful_optimizer_relations_ = false;
    partitioned_nodes_ = false;

    train_ = train;
    data_set_type_ = data_set_type;

    storage_ptrs_ = storage_ptrs;

    if (storage_ptrs_.edge_features != nullptr) {
        featurized_edges_ = true;
    }

    if (storage_ptrs_.node_feature_storage != nullptr) {
        featurized_nodes_ = true;
    }

    if (storage_ptrs_.relation_feature_storage != nullptr) {
        featurized_relations_ = true;
    }

    if (storage_ptrs_.node_optimizer_state != nullptr) {
        stateful_optimizer_nodes_ = true;
    }

    if (storage_ptrs_.relations_optimizer_state != nullptr) {
        stateful_optimizer_relations_ = true;
    }

    if (storage_options.num_partitions > 1) {
        partitioned_nodes_ = true;
    }
}

GraphModelStorage::~GraphStorage() {
    delete storage_ptrs_.edges;
    delete storage_ptrs_.edge_features;
    delete storage_ptrs_.node_embedding_storage;
    delete storage_ptrs_.node_optimizer_state;
    delete storage_ptrs_.node_feature_storage;
    delete storage_ptrs_.lhs_relations_storage;
    delete storage_ptrs_.lhs_relations_optimizer_state;
    delete storage_ptrs_.rhs_relations_storage;
    delete storage_ptrs_.rhs_relations_optimizer_state;
    delete storage_ptrs_.rhs_relations_feature_storage;
    delete storage_ptrs_.relations_storage;
    delete storage_ptrs_.relations_optimizer_state;
    delete storage_ptrs_.relation_feature_storage;
}

void GraphModelStorage::setEdgesStorage(Storage *edge_storage) {
    storage_ptrs_.edges = edge_storage;
}

EdgeList GraphModelStorage::getEdges(Indices indices) {
    return storage_ptrs_.edges->indexRead(indices);
}

EdgeList GraphModelStorage::getEdgesRange(int64_t start, int64_t size) {
    return storage_ptrs_.edges->range(start, size);
}

void GraphModelStorage::updateEdges(Indices indices, EdgeList edges) {
    storage_ptrs_.edges->indexPut(indices, edges);
}

void GraphModelStorage::shuffleEdges() {
    storage_ptrs_.edges->shuffle();
}

Embeddings GraphModelStorage::getNodeEmbeddings(Indices indices) {
    return storage_ptrs_.node_embedding_storage->indexRead(indices);
}

Embeddings GraphModelStorage::getNodeEmbeddingsRange(int64_t start, int64_t size) {
    return storage_ptrs_.node_embedding_storage->range(start, size);
}

void GraphModelStorage::updatePutNodeEmbeddings(Indices indices, Embeddings embeddings) {
    storage_ptrs_.node_embedding_storage->indexPut(indices, embeddings);
}

void GraphModelStorage::updateAddNodeEmbeddings(Indices indices, torch::Tensor values) {
    storage_ptrs_.node_embedding_storage->indexAdd(indices, values);
}

OptimizerState GraphModelStorage::getNodeEmbeddingState(Indices indices) {
    return storage_ptrs_.node_optimizer_state->indexRead(indices);
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateRange(int64_t start, int64_t size) {
    return storage_ptrs_.node_optimizer_state->range(start, size);
}

void GraphModelStorage::updatePutNodeEmbeddingState(Indices indices, OptimizerState state) {
    storage_ptrs_.node_optimizer_state->indexPut(indices, state);
}

void GraphModelStorage::updateAddNodeEmbeddingState(Indices indices, torch::Tensor values) {
    storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
}

Relations GraphModelStorage::getRelations(Indices indices, bool lhs) {
    if (lhs) {
        return storage_ptrs_.lhs_relations_storage->indexRead(indices);
    } else {
        return storage_ptrs_.rhs_relations_storage->indexRead(indices);
    }
}

Relations GraphModelStorage::getRelationsRange(int64_t start, int64_t size, bool lhs) {
    if (lhs) {
        return storage_ptrs_.lhs_relations_storage->range(start, size);
    } else {
        return storage_ptrs_.rhs_relations_storage->range(start, size);
    }
}

void GraphModelStorage::updatePutRelations(Indices indices, Relations embeddings, bool lhs) {
    if (lhs) {
        storage_ptrs_.lhs_relations_storage->indexPut(indices, embeddings);
    } else {
        storage_ptrs_.rhs_relations_storage->indexPut(indices, embeddings);
    }
}

void GraphModelStorage::updateAddRelations(Indices indices, torch::Tensor values, bool lhs) {
    if (lhs) {
        storage_ptrs_.lhs_relations_storage->indexAdd(indices, values);
    } else {
        storage_ptrs_.rhs_relations_storage->indexAdd(indices, values);
    }
}

OptimizerState GraphModelStorage::getRelationsState(Indices indices, bool lhs) {
    if (lhs) {
        return storage_ptrs_.lhs_relations_optimizer_state->indexRead(indices);
    } else {
        return storage_ptrs_.rhs_relations_optimizer_state->indexRead(indices);
    }
}

OptimizerState GraphModelStorage::getRelationsStateRange(int64_t start, int64_t size, bool lhs) {
    if (lhs) {
        return storage_ptrs_.lhs_relations_optimizer_state->range(start, size);
    } else {
        return storage_ptrs_.rhs_relations_optimizer_state->range(start, size);
    }
}

void GraphModelStorage::updatePutRelationsState(Indices indices, OptimizerState state, bool lhs) {
    if (lhs) {
        storage_ptrs_.lhs_relations_optimizer_state->indexPut(indices, state);
    } else {
        storage_ptrs_.rhs_relations_optimizer_state->indexPut(indices, state);
    }
}

void GraphModelStorage::updateAddRelationsState(Indices indices, torch::Tensor values, bool lhs) {
    if (lhs) {
        storage_ptrs_.lhs_relations_optimizer_state->indexAdd(indices, values);
    } else {
        storage_ptrs_.rhs_relations_optimizer_state->indexAdd(indices, values);
    }
}