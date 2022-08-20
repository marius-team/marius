//
// Created by jasonmohoney on 10/4/19.
//

#include "io.h"

#include "configuration/constants.h"
#include "initialization.h"
#include "logger.h"
#include "model.h"

std::tuple<Storage *, Storage *, Storage *, Storage *> initializeEdges(shared_ptr<StorageConfig> storage_config, LearningTask learning_task) {

    string train_filename = storage_config->dataset->base_directory
        + PathConstants::edges_directory
        + PathConstants::training
        + PathConstants::edges_file
        + PathConstants::file_ext;
    string valid_filename = storage_config->dataset->base_directory
        + PathConstants::edges_directory
        + PathConstants::validation
        + PathConstants::edges_file
        + PathConstants::file_ext;
    string test_filename = storage_config->dataset->base_directory
        + PathConstants::edges_directory
        + PathConstants::test
        + PathConstants::edges_file
        + PathConstants::file_ext;

    // Note: may want this as output of preprocessing in case this copy isn't sufficient
    string train_dst_sort_filename = storage_config->dataset->base_directory
                                     + PathConstants::edges_directory
                                     + PathConstants::training
                                     + PathConstants::edges_file
                                     + "_dst_sort"
                                     + PathConstants::file_ext;
//    copyFile(train_filename, train_dst_sort_filename);

    Storage *train_edge_storage;
    Storage *train_edge_storage_dst_sort;
    Storage *valid_edge_storage;
    Storage *test_edge_storage;

    int64_t num_train = 0;
    int64_t num_valid = 0;
    int64_t num_test = 0;
    if (learning_task == LearningTask::LINK_PREDICTION) {
        num_train = storage_config->dataset->num_train;
        num_valid = storage_config->dataset->num_valid;
        num_test = storage_config->dataset->num_test;
    } else if (learning_task == LearningTask::NODE_CLASSIFICATION) {
        num_train = storage_config->dataset->num_edges;
    }

    torch::Dtype dtype = storage_config->edges->options->dtype;

    int num_columns = 3;
    if (storage_config->dataset->num_relations == 1) {
        num_columns = 2;
    }

    switch (storage_config->edges->type) {
        case StorageBackend::PARTITION_BUFFER: {
            SPDLOG_ERROR("Backend type not available for edges.");
            throw std::runtime_error("");
        }
        case StorageBackend::FLAT_FILE: {
            train_edge_storage = new FlatFile(train_filename, num_train, num_columns, dtype);
//            train_edge_storage_dst_sort = new FlatFile(train_dst_sort_filename, num_train, num_columns, dtype);
            valid_edge_storage = new FlatFile(valid_filename, num_valid, num_columns, dtype);
            test_edge_storage = new FlatFile(test_filename, num_test, num_columns, dtype);
            break;
        }
        case StorageBackend::HOST_MEMORY: {
            train_edge_storage = new InMemory(train_filename, num_train, num_columns, dtype, torch::kCPU);
//            train_edge_storage_dst_sort = new InMemory(train_dst_sort_filename, num_train, num_columns, dtype, torch::kCPU);
            valid_edge_storage = new InMemory(valid_filename, num_valid, num_columns, dtype, torch::kCPU);
            test_edge_storage = new InMemory(test_filename, num_test, num_columns, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            train_edge_storage = new InMemory(train_filename, num_train, num_columns, dtype, storage_config->device_type);
//            train_edge_storage_dst_sort = new InMemory(train_dst_sort_filename, num_train, num_columns, dtype, storage_config->device_type);
            valid_edge_storage = new InMemory(valid_filename, num_valid, num_columns, dtype, storage_config->device_type);
            test_edge_storage = new InMemory(test_filename, num_test, num_columns, dtype, storage_config->device_type);
            break;
        }
    }

    bool use_buffer = false;

    if (storage_config->embeddings != nullptr) {
        if (storage_config->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            use_buffer = true;
        }
    }

    if (storage_config->features != nullptr) {
        if (storage_config->features->type == StorageBackend::PARTITION_BUFFER) {
            use_buffer = true;
        }
    }

    if (use_buffer) {
        string train_edges_partitions = storage_config->dataset->base_directory
                                        + PathConstants::edges_directory
                                        + PathConstants::training
                                        + PathConstants::edge_partition_offsets_file;

        string validation_edges_partitions = storage_config->dataset->base_directory
                                             + PathConstants::edges_directory
                                             + PathConstants::validation
                                             + PathConstants::edge_partition_offsets_file;

        string test_edges_partitions = storage_config->dataset->base_directory
                                       + PathConstants::edges_directory
                                       + PathConstants::test
                                       + PathConstants::edge_partition_offsets_file;

        train_edge_storage->readPartitionSizes(train_edges_partitions);
//        train_edge_storage_dst_sort->readPartitionSizes(train_edges_partitions);
        valid_edge_storage->readPartitionSizes(validation_edges_partitions);
        test_edge_storage->readPartitionSizes(test_edges_partitions);
    }

//    train_edge_storage->sort(true);
//    train_edge_storage_dst_sort->sort(false);

    if (storage_config->shuffle_input) {
//        train_edge_storage->shuffle();
        valid_edge_storage->shuffle();
        test_edge_storage->shuffle();
    }

//    return std::forward_as_tuple(train_edge_storage, train_edge_storage_dst_sort, valid_edge_storage, test_edge_storage);
    return std::forward_as_tuple(train_edge_storage, nullptr, valid_edge_storage, test_edge_storage);
}

std::tuple<Storage *, Storage *> initializeNodeEmbeddings(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {

    string node_embedding_filename = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::embeddings_file
        + PathConstants::file_ext;
    string optimizer_state_filename = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::embeddings_state_file
        + PathConstants::file_ext;

    if (storage_config->embeddings == nullptr) {
        return std::forward_as_tuple(nullptr, nullptr);
    }

    int64_t num_nodes = storage_config->dataset->num_nodes;
    int embedding_dim = model->model_config_->embeddings->dimension;
    torch::Dtype dtype = storage_config->embeddings->options->dtype;

    if (model->reinitialize_) {
        FlatFile *init_node_embeddings = new FlatFile(node_embedding_filename, dtype);
        FlatFile *init_optimizer_state_storage = new FlatFile(optimizer_state_filename, dtype);

        int64_t curr_num_nodes = 0;
        int64_t offset = 0;

        while (offset < num_nodes) {
            if (num_nodes - offset < MAX_NODE_EMBEDDING_INIT_SIZE) {
                curr_num_nodes = num_nodes - offset;
            } else {
                curr_num_nodes = MAX_NODE_EMBEDDING_INIT_SIZE;
            }

            Embeddings weights = initialize_subtensor(model->model_config_->embeddings->init,
                                                      {curr_num_nodes, embedding_dim},
                                                      {num_nodes, embedding_dim},
                                                      torch::TensorOptions());
            OptimizerState emb_state = torch::zeros_like(weights);
            init_node_embeddings->append(weights);
            init_optimizer_state_storage->append(emb_state);

            offset += curr_num_nodes;
        }

        delete init_node_embeddings;
        delete init_optimizer_state_storage;
    }

    Storage *node_embeddings;
    Storage *optimizer_state_storage;

    switch (storage_config->embeddings->type) {
        case StorageBackend::PARTITION_BUFFER: {
            node_embeddings = new PartitionBufferStorage(node_embedding_filename,
                                                         num_nodes,
                                                         embedding_dim,
                                                         std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->embeddings->options));
            if (model->train_) {
                optimizer_state_storage = new PartitionBufferStorage(optimizer_state_filename,
                                                                     num_nodes,
                                                                     embedding_dim,
                                                                     std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->embeddings->options));
            }
            break;
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            node_embeddings = new InMemory(node_embedding_filename,
                                           num_nodes,
                                           embedding_dim,
                                           dtype,
                                           torch::kCPU);
            if (model->train_) {
                optimizer_state_storage = new InMemory(optimizer_state_filename,
                                                       num_nodes,
                                                       embedding_dim,
                                                       dtype,
                                                       torch::kCPU);
            }
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_embeddings = new InMemory(node_embedding_filename,
                                           num_nodes,
                                           embedding_dim,
                                           dtype,
                                           storage_config->device_type);
            if (model->train_) {
                optimizer_state_storage = new InMemory(optimizer_state_filename,
                                                       num_nodes,
                                                       embedding_dim,
                                                       dtype,
                                                       storage_config->device_type);
            }
            break;
        }
    }

    return std::forward_as_tuple(node_embeddings, optimizer_state_storage);
}

std::tuple<Storage *, Storage *, Storage *> initializeNodeIds(shared_ptr<StorageConfig> storage_config) {

    string train_filename = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::training
        + PathConstants::nodes_file
        + PathConstants::file_ext;
    string valid_filename = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::validation
        + PathConstants::nodes_file
        + PathConstants::file_ext;
    string test_filename = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::test
        + PathConstants::nodes_file
        + PathConstants::file_ext;

    int64_t num_train = storage_config->dataset->num_train;
    int64_t num_valid = storage_config->dataset->num_valid;
    int64_t num_test = storage_config->dataset->num_test;
    torch::Dtype dtype = storage_config->nodes->options->dtype;

    Storage *train_node_storage;
    Storage *valid_node_storage;
    Storage *test_node_storage;

    switch (storage_config->nodes->type) {
        case StorageBackend::PARTITION_BUFFER: {
            SPDLOG_ERROR("Backend type not available for nodes.");
            throw std::runtime_error("");
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for nodes.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            train_node_storage = new InMemory(train_filename, num_train, 1, dtype, torch::kCPU);
            valid_node_storage = new InMemory(valid_filename, num_valid, 1, dtype, torch::kCPU);
            test_node_storage = new InMemory(test_filename, num_test, 1, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            train_node_storage = new InMemory(train_filename, num_train, 1, dtype, storage_config->device_type);
            valid_node_storage = new InMemory(valid_filename, num_valid, 1, dtype, storage_config->device_type);
            test_node_storage = new InMemory(test_filename, num_test, 1, dtype, storage_config->device_type);
            break;
        }
    }

    if (storage_config->shuffle_input) {
        train_node_storage->shuffle();
        valid_node_storage->shuffle();
        test_node_storage->shuffle();
    }

    return std::forward_as_tuple(train_node_storage, valid_node_storage, test_node_storage);
}

Storage *initializeNodeFeatures(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {

    string node_features_file = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::features_file
        + PathConstants::file_ext;

    Storage *node_features;

    int64_t num_nodes = storage_config->dataset->num_nodes;
    int64_t feature_dim = storage_config->dataset->feature_dim;

    if (storage_config->features == nullptr) {
        return nullptr;
    }
    torch::Dtype dtype = storage_config->features->options->dtype;

    switch (storage_config->features->type) {
        case StorageBackend::PARTITION_BUFFER: {
            node_features = new PartitionBufferStorage(node_features_file,
                                                       num_nodes,
                                                       feature_dim,
                                                       std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->features->options));
            break;
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for features.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            node_features = new InMemory(node_features_file, num_nodes, feature_dim, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_features = new InMemory(node_features_file, num_nodes, feature_dim, dtype, storage_config->device_type);
            break;
        }
    }

    return node_features;
}

Storage *initializeNodeLabels(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {

    string node_labels_file = storage_config->dataset->base_directory
        + PathConstants::nodes_directory
        + PathConstants::labels_file
        + PathConstants::file_ext;


    Storage *node_labels;

    int64_t num_nodes = storage_config->dataset->num_nodes;
    int64_t num_classes = storage_config->dataset->num_classes;
    torch::Dtype dtype = torch::kInt32;

    switch (storage_config->nodes->type) {
        case StorageBackend::PARTITION_BUFFER: {
            SPDLOG_ERROR("Backend type not available for nodes/labels.");
            throw std::runtime_error("");
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for nodes/labels.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            node_labels = new InMemory(node_labels_file, num_nodes, 1, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_labels = new InMemory(node_labels_file, num_nodes, 1, dtype, storage_config->device_type);
            break;
        }
    }

    return node_labels;
}

GraphModelStorage *initializeStorageLinkPrediction(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {

    std::tuple<Storage *, Storage *, Storage *, Storage *> edge_storages = initializeEdges( storage_config, model->learning_task_);
    std::tuple<Storage *, Storage *> node_embeddings = initializeNodeEmbeddings(model, storage_config);

    GraphModelStoragePtrs storage_ptrs = {};

    storage_ptrs.train_edges = std::get<0>(edge_storages);
    storage_ptrs.train_edges_dst_sort = std::get<1>(edge_storages);
    storage_ptrs.validation_edges = std::get<2>(edge_storages);
    storage_ptrs.test_edges = std::get<3>(edge_storages);

    storage_ptrs.node_features = initializeNodeFeatures(model, storage_config);
    storage_ptrs.node_embeddings = std::get<0>(node_embeddings);
    storage_ptrs.node_optimizer_state = std::get<1>(node_embeddings);

    GraphModelStorage *graph_model_storage = new GraphModelStorage(storage_ptrs, storage_config, model->learning_task_, model->filtered_eval_);

    return graph_model_storage;
}

GraphModelStorage *initializeStorageNodeClassification(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {

    std::tuple<Storage *, Storage *, Storage *, Storage *> edge_storages = initializeEdges(storage_config, model->learning_task_);
    std::tuple<Storage *, Storage *, Storage *> node_id_storages = initializeNodeIds(storage_config);
    Storage *node_features = initializeNodeFeatures(model, storage_config);
    Storage *node_labels = initializeNodeLabels(model, storage_config);

    GraphModelStoragePtrs storage_ptrs = {};

    storage_ptrs.train_edges = std::get<0>(edge_storages);
    storage_ptrs.train_edges_dst_sort = std::get<1>(edge_storages);
    storage_ptrs.edges = storage_ptrs.train_edges;

    storage_ptrs.train_nodes = std::get<0>(node_id_storages);
    storage_ptrs.valid_nodes = std::get<1>(node_id_storages);
    storage_ptrs.test_nodes = std::get<2>(node_id_storages);

    storage_ptrs.nodes = storage_ptrs.train_nodes;
    storage_ptrs.node_features = node_features;
    storage_ptrs.node_labels = node_labels;

    std::tuple<Storage *, Storage *> node_embeddings = initializeNodeEmbeddings(model, storage_config);
    storage_ptrs.node_embeddings = std::get<0>(node_embeddings);
    storage_ptrs.node_optimizer_state = std::get<1>(node_embeddings);

    GraphModelStorage *graph_model_storage = new GraphModelStorage(storage_ptrs, storage_config, model->learning_task_, model->filtered_eval_);

    return graph_model_storage;
}

GraphModelStorage *initializeStorage(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {
    if (model->learning_task_ == LearningTask::LINK_PREDICTION) {
        return initializeStorageLinkPrediction(model, storage_config);
    } else if (model->learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        return initializeStorageNodeClassification(model, storage_config);
    } else {
        SPDLOG_ERROR("Unsupported Learning Task");
        throw std::runtime_error("");
    }
}

GraphModelStorage *initializeTrainEdgesStorage(shared_ptr<StorageConfig> storage_config) {
    std::tuple<Storage *, Storage *, Storage *, Storage *> edge_storages = initializeEdges(storage_config, LearningTask::NODE_CLASSIFICATION);
    GraphModelStoragePtrs storage_ptrs = {};
    storage_ptrs.train_edges = std::get<0>(edge_storages);
    storage_ptrs.train_edges_dst_sort = std::get<1>(edge_storages);

    GraphModelStorage *graph_model_storage = new GraphModelStorage(storage_ptrs, storage_config);

    return graph_model_storage;
}
