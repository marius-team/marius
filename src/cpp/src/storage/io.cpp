//
// Created by jasonmohoney on 10/4/19.
//

#include "storage/io.h"

#include "configuration/constants.h"
#include "nn/initialization.h"
#include "nn/model.h"
#include "reporting/logger.h"

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> initializeEdges(shared_ptr<StorageConfig> storage_config,
                                                                                                               LearningTask learning_task) {
    string train_filename =
        storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::training + PathConstants::edges_file + PathConstants::file_ext;
    string valid_filename =
        storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::validation + PathConstants::edges_file + PathConstants::file_ext;
    string test_filename =
        storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::test + PathConstants::edges_file + PathConstants::file_ext;

    string train_dst_sort_filename = storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::training +
                                     PathConstants::edges_file + PathConstants::dst_sort + PathConstants::file_ext;

    shared_ptr<Storage> train_edge_storage = nullptr;
    shared_ptr<Storage> train_edge_storage_dst_sort = nullptr;
    shared_ptr<Storage> valid_edge_storage = nullptr;
    shared_ptr<Storage> test_edge_storage = nullptr;

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
            if (num_train != -1) {
                train_edge_storage = std::make_shared<FlatFile>(train_filename, num_train, num_columns, dtype);
            }
            if (num_valid != -1) {
                valid_edge_storage = std::make_shared<FlatFile>(valid_filename, num_valid, num_columns, dtype);
            }
            if (num_test != -1) {
                test_edge_storage = std::make_shared<FlatFile>(test_filename, num_test, num_columns, dtype);
            }
            break;
        }
        case StorageBackend::HOST_MEMORY: {
            if (num_train != -1) {
                train_edge_storage = std::make_shared<InMemory>(train_filename, num_train, num_columns, dtype, torch::kCPU);
                if (!storage_config->train_edges_pre_sorted) {
                    copyFile(train_filename, train_dst_sort_filename);
                }
                train_edge_storage_dst_sort = std::make_shared<InMemory>(train_dst_sort_filename, num_train, num_columns, dtype, torch::kCPU);
            }
            if (num_valid != -1) {
                valid_edge_storage = std::make_shared<InMemory>(valid_filename, num_valid, num_columns, dtype, torch::kCPU);
            }
            if (num_test != -1) {
                test_edge_storage = std::make_shared<InMemory>(test_filename, num_test, num_columns, dtype, torch::kCPU);
            }
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            if (num_train != -1) {
                train_edge_storage = std::make_shared<InMemory>(train_filename, num_train, num_columns, dtype, storage_config->device_type);
                if (!storage_config->train_edges_pre_sorted) {
                    copyFile(train_filename, train_dst_sort_filename);
                }
                train_edge_storage_dst_sort = std::make_shared<InMemory>(train_dst_sort_filename, num_train, num_columns, dtype, storage_config->device_type);
            }
            if (num_valid != -1) {
                valid_edge_storage = std::make_shared<InMemory>(valid_filename, num_valid, num_columns, dtype, storage_config->device_type);
            }
            if (num_test != -1) {
                test_edge_storage = std::make_shared<InMemory>(test_filename, num_test, num_columns, dtype, storage_config->device_type);
            }
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
        string train_edges_partitions =
            storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::training + PathConstants::edge_partition_offsets_file;

        string validation_edges_partitions =
            storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::validation + PathConstants::edge_partition_offsets_file;

        string test_edges_partitions =
            storage_config->dataset->dataset_dir + PathConstants::edges_directory + PathConstants::test + PathConstants::edge_partition_offsets_file;

        if (train_edge_storage != nullptr) {
            train_edge_storage->readPartitionSizes(train_edges_partitions);
        }

        if (valid_edge_storage != nullptr) {
            valid_edge_storage->readPartitionSizes(validation_edges_partitions);
        }

        if (test_edge_storage != nullptr) {
            test_edge_storage->readPartitionSizes(test_edges_partitions);
        }
    } else {
        if (train_edge_storage != nullptr) {
            if (!storage_config->train_edges_pre_sorted) {
                train_edge_storage->sort(true);
                train_edge_storage_dst_sort->sort(false);
            }
        }
    }

    if (storage_config->shuffle_input) {
        if (valid_edge_storage != nullptr) {
            valid_edge_storage->shuffle();
        }
        if (test_edge_storage != nullptr) {
            test_edge_storage->shuffle();
        }
    }

    return std::forward_as_tuple(train_edge_storage, train_edge_storage_dst_sort, valid_edge_storage, test_edge_storage);
}

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>> initializeNodeEmbeddings(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config,
                                                                              bool reinitialize, bool train, shared_ptr<InitConfig> init_config) {
    string node_embedding_filename = storage_config->model_dir + PathConstants::embeddings_file + PathConstants::file_ext;
    string optimizer_state_filename = storage_config->model_dir + PathConstants::embeddings_state_file + PathConstants::file_ext;

    if (storage_config->embeddings == nullptr || !model->has_embeddings()) {
        return std::forward_as_tuple(nullptr, nullptr);
    }

    int64_t num_nodes = storage_config->dataset->num_nodes;
    int embedding_dim = model->get_base_embedding_dim();
    torch::Dtype dtype = storage_config->embeddings->options->dtype;

    if (reinitialize) {
        shared_ptr<FlatFile> init_node_embeddings = std::make_shared<FlatFile>(node_embedding_filename, dtype);
        shared_ptr<FlatFile> init_optimizer_state_storage = std::make_shared<FlatFile>(optimizer_state_filename, dtype);

        int64_t curr_num_nodes = 0;
        int64_t offset = 0;

        while (offset < num_nodes) {
            if (num_nodes - offset < MAX_NODE_EMBEDDING_INIT_SIZE) {
                curr_num_nodes = num_nodes - offset;
            } else {
                curr_num_nodes = MAX_NODE_EMBEDDING_INIT_SIZE;
            }

            torch::Tensor weights = initialize_subtensor(init_config, {curr_num_nodes, embedding_dim}, {num_nodes, embedding_dim}, torch::TensorOptions());
            OptimizerState emb_state = torch::zeros_like(weights);
            init_node_embeddings->append(weights);
            init_optimizer_state_storage->append(emb_state);

            offset += curr_num_nodes;
        }
    }

    shared_ptr<Storage> node_embeddings = nullptr;
    shared_ptr<Storage> optimizer_state_storage = nullptr;

    switch (storage_config->embeddings->type) {
        case StorageBackend::PARTITION_BUFFER: {
            node_embeddings = std::make_shared<PartitionBufferStorage>(node_embedding_filename, num_nodes, embedding_dim,
                                                                       std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->embeddings->options));
            if (train) {
                optimizer_state_storage = std::make_shared<PartitionBufferStorage>(
                    optimizer_state_filename, num_nodes, embedding_dim, std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->embeddings->options));
            }
            break;
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            node_embeddings = std::make_shared<InMemory>(node_embedding_filename, num_nodes, embedding_dim, dtype, torch::kCPU);
            if (train) {
                optimizer_state_storage = std::make_shared<InMemory>(optimizer_state_filename, num_nodes, embedding_dim, dtype, torch::kCPU);
            }
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_embeddings = std::make_shared<InMemory>(node_embedding_filename, num_nodes, embedding_dim, dtype, storage_config->device_type);
            if (train) {
                optimizer_state_storage = std::make_shared<InMemory>(optimizer_state_filename, num_nodes, embedding_dim, dtype, storage_config->device_type);
            }
            break;
        }
    }

    return std::forward_as_tuple(node_embeddings, optimizer_state_storage);
}

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> initializeNodeIds(shared_ptr<StorageConfig> storage_config) {
    string train_filename =
        storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::training + PathConstants::nodes_file + PathConstants::file_ext;
    string valid_filename =
        storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::validation + PathConstants::nodes_file + PathConstants::file_ext;
    string test_filename =
        storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::test + PathConstants::nodes_file + PathConstants::file_ext;

    int64_t num_train = storage_config->dataset->num_train;
    int64_t num_valid = storage_config->dataset->num_valid;
    int64_t num_test = storage_config->dataset->num_test;
    torch::Dtype dtype = storage_config->nodes->options->dtype;

    shared_ptr<Storage> train_node_storage = nullptr;
    shared_ptr<Storage> valid_node_storage = nullptr;
    shared_ptr<Storage> test_node_storage = nullptr;

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
            if (num_train != -1) {
                train_node_storage = std::make_shared<InMemory>(train_filename, num_train, 1, dtype, torch::kCPU);
            }
            if (num_valid != -1) {
                valid_node_storage = std::make_shared<InMemory>(valid_filename, num_valid, 1, dtype, torch::kCPU);
            }
            if (num_test != -1) {
                test_node_storage = std::make_shared<InMemory>(test_filename, num_test, 1, dtype, torch::kCPU);
            }

            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            if (num_train != -1) {
                train_node_storage = std::make_shared<InMemory>(train_filename, num_train, 1, dtype, storage_config->device_type);
            }
            if (num_valid != -1) {
                valid_node_storage = std::make_shared<InMemory>(valid_filename, num_valid, 1, dtype, storage_config->device_type);
            }
            if (num_test != -1) {
                test_node_storage = std::make_shared<InMemory>(test_filename, num_test, 1, dtype, storage_config->device_type);
            }
            break;
        }
    }

    if (storage_config->shuffle_input) {
        if (train_node_storage != nullptr) {
            train_node_storage->shuffle();
        }
        if (valid_node_storage != nullptr) {
            valid_node_storage->shuffle();
        }
        if (test_node_storage != nullptr) {
            test_node_storage->shuffle();
        }
    }

    return std::forward_as_tuple(train_node_storage, valid_node_storage, test_node_storage);
}

shared_ptr<Storage> initializeRelationFeatures(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {
    string rel_features_file = storage_config->model_dir + PathConstants::features_file + PathConstants::file_ext;

    int64_t num_relations = storage_config->dataset->num_relations;
    int64_t rel_feature_dim = storage_config->dataset->rel_feature_dim;

    if (rel_feature_dim == -1 || num_relations == -1 || model->decoder_ == nullptr) {
        return nullptr;
    }

    shared_ptr<Storage> rel_features =
        std::make_shared<InMemory>(rel_features_file, num_relations, rel_feature_dim, torch::kFloat32, storage_config->device_type);
    rel_features->load();

    return rel_features;
}

shared_ptr<Storage> initializeNodeFeatures(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {
    string node_features_file = storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::features_file + PathConstants::file_ext;

    shared_ptr<Storage> node_features;

    int64_t num_nodes = storage_config->dataset->num_nodes;
    int64_t node_feature_dim = storage_config->dataset->node_feature_dim;

    if (storage_config->features == nullptr || node_feature_dim == -1) {
        return nullptr;
    }
    torch::Dtype dtype = storage_config->features->options->dtype;

    switch (storage_config->features->type) {
        case StorageBackend::PARTITION_BUFFER: {
            node_features = std::make_shared<PartitionBufferStorage>(node_features_file, num_nodes, node_feature_dim,
                                                                     std::dynamic_pointer_cast<PartitionBufferOptions>(storage_config->features->options));
            break;
        }
        case StorageBackend::FLAT_FILE: {
            SPDLOG_ERROR("Backend type not available for features.");
            throw std::runtime_error("");
        }
        case StorageBackend::HOST_MEMORY: {
            node_features = std::make_shared<InMemory>(node_features_file, num_nodes, node_feature_dim, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_features = std::make_shared<InMemory>(node_features_file, num_nodes, node_feature_dim, dtype, storage_config->device_type);
            break;
        }
    }

    return node_features;
}

shared_ptr<Storage> initializeNodeLabels(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config) {
    string node_labels_file = storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::labels_file + PathConstants::file_ext;

    shared_ptr<Storage> node_labels;

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
            node_labels = std::make_shared<InMemory>(node_labels_file, num_nodes, 1, dtype, torch::kCPU);
            break;
        }
        case StorageBackend::DEVICE_MEMORY: {
            node_labels = std::make_shared<InMemory>(node_labels_file, num_nodes, 1, dtype, storage_config->device_type);
            break;
        }
    }

    return node_labels;
}

shared_ptr<GraphModelStorage> initializeStorageLinkPrediction(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize, bool train,
                                                              shared_ptr<InitConfig> init_config) {
    std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> edge_storages =
        initializeEdges(storage_config, model->learning_task_);
    std::tuple<shared_ptr<Storage>, shared_ptr<Storage>> node_embeddings = initializeNodeEmbeddings(model, storage_config, reinitialize, train, init_config);

    GraphModelStoragePtrs storage_ptrs = {};

    storage_ptrs.train_edges = std::get<0>(edge_storages);
    storage_ptrs.train_edges_dst_sort = std::get<1>(edge_storages);
    storage_ptrs.validation_edges = std::get<2>(edge_storages);
    storage_ptrs.test_edges = std::get<3>(edge_storages);

    storage_ptrs.node_features = initializeNodeFeatures(model, storage_config);
    storage_ptrs.node_embeddings = std::get<0>(node_embeddings);
    storage_ptrs.node_optimizer_state = std::get<1>(node_embeddings);

    storage_ptrs.relation_features = initializeRelationFeatures(model, storage_config);

    shared_ptr<GraphModelStorage> graph_model_storage = std::make_shared<GraphModelStorage>(storage_ptrs, storage_config);

    return graph_model_storage;
}

shared_ptr<GraphModelStorage> initializeStorageNodeClassification(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize,
                                                                  bool train, shared_ptr<InitConfig> init_config) {
    std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> edge_storages =
        initializeEdges(storage_config, model->learning_task_);
    std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> node_id_storages = initializeNodeIds(storage_config);
    shared_ptr<Storage> node_features = initializeNodeFeatures(model, storage_config);
    shared_ptr<Storage> node_labels = initializeNodeLabels(model, storage_config);

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

    std::tuple<shared_ptr<Storage>, shared_ptr<Storage>> node_embeddings = initializeNodeEmbeddings(model, storage_config, reinitialize, train, init_config);
    storage_ptrs.node_embeddings = std::get<0>(node_embeddings);
    storage_ptrs.node_optimizer_state = std::get<1>(node_embeddings);

    shared_ptr<GraphModelStorage> graph_model_storage = std::make_shared<GraphModelStorage>(storage_ptrs, storage_config);

    return graph_model_storage;
}

shared_ptr<GraphModelStorage> initializeStorage(shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize, bool train,
                                                shared_ptr<InitConfig> init_config) {
    if (init_config == nullptr) {
        init_config = std::make_shared<InitConfig>();
        init_config->type = InitDistribution::GLOROT_UNIFORM;
    }

    if (model->learning_task_ == LearningTask::LINK_PREDICTION) {
        return initializeStorageLinkPrediction(model, storage_config, reinitialize, train, init_config);
    } else if (model->learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        return initializeStorageNodeClassification(model, storage_config, reinitialize, train, init_config);
    } else {
        SPDLOG_ERROR("Unsupported Learning Task");
        throw std::runtime_error("");
    }
}