//
// Created by jasonmohoney on 10/4/19.
//

#include "io.h"

#include <filesystem>

#include "config.h"
#include "logger.h"

using std::get;
using std::ofstream;
using std::map;
using std::vector;
using std::tuple;
using std::mutex;
using std::pair;
using std::forward_as_tuple;
using std::ios;
using std::to_string;
using std::cout;

void createDir(const string &path) {

    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
            SPDLOG_DEBUG("{} directory already exists", path);
        } else {
            SPDLOG_ERROR("Failed to create {}\nError: {}", path, errno);
        }
    }
}

void initOutputDir(const string &output_directory) {

    createDir(marius_options.path.base_directory);

    createDir(output_directory);

    createDir(output_directory + PathConstants::embeddings_directory);

    createDir(output_directory + PathConstants::relations_directory);

    createDir(output_directory + PathConstants::edges_directory);

    createDir(output_directory + PathConstants::edges_directory + PathConstants::edges_train_directory);

    createDir(output_directory + PathConstants::edges_directory + PathConstants::edges_validation_directory);

    createDir(output_directory + PathConstants::edges_directory + PathConstants::edges_test_directory);
}

tuple<Storage *, Storage *, Storage *> initializeEdges(bool train) {
    int64_t num_train = marius_options.general.num_train;
    int64_t num_valid = marius_options.general.num_valid;
    int64_t num_test = marius_options.general.num_test;

    bool reinitialize = marius_options.storage.reinitialize_edges;

    // if reinitialize is false but the edge files aren't present, we must reinitialize.

    string train_filename = marius_options.path.experiment_directory
        + PathConstants::edges_directory
        + PathConstants::edges_train_directory
        + PathConstants::edges_file
        + PathConstants::file_ext;
    string valid_filename = marius_options.path.experiment_directory
        + PathConstants::edges_directory
        + PathConstants::edges_validation_directory
        + PathConstants::edges_file
        + PathConstants::file_ext;
    string test_filename = marius_options.path.experiment_directory
        + PathConstants::edges_directory
        + PathConstants::edges_test_directory
        + PathConstants::edges_file
        + PathConstants::file_ext;

    Storage *train_edge_storage;
    Storage *valid_edge_storage;
    Storage *test_edge_storage;

    if (reinitialize && train) {
        string input_train_filename = marius_options.path.train_edges;
        string input_valid_filename = marius_options.path.validation_edges; // might be empty
        string input_test_filename = marius_options.path.test_edges; // might be empty

        FlatFile *input_train_file = new FlatFile(input_train_filename, num_train, 3, marius_options.storage.edges_dtype);
        if (marius_options.storage.remove_preprocessed) {
            input_train_file->move(train_filename);
        } else {
            input_train_file->copy(train_filename, false);
        }
        delete input_train_file;

        if (!input_valid_filename.empty()) {
            FlatFile *input_valid_file = new FlatFile(input_valid_filename, num_valid, 3, marius_options.storage.edges_dtype);
            if (marius_options.storage.remove_preprocessed) {
                input_valid_file->move(valid_filename);
            } else {
                input_valid_file->copy(valid_filename, false);
            }
            delete input_valid_file;
        }

        if (!input_test_filename.empty()) {
            FlatFile *input_test_file = new FlatFile(input_test_filename, num_test, 3, marius_options.storage.edges_dtype);
            if (marius_options.storage.remove_preprocessed) {
                input_test_file->move(test_filename);
            } else {
                input_test_file->copy(test_filename, false);
            }
            delete input_test_file;
        }
    }

    switch (marius_options.storage.edges) {
        case BackendType::RocksDB: {
            SPDLOG_ERROR("Currently Unsupported");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            SPDLOG_ERROR("Backend type not available for edges.");
            exit(-1);
        }
        case BackendType::FlatFile: {
            train_edge_storage = new FlatFile(train_filename, num_train, 3, marius_options.storage.edges_dtype);
            valid_edge_storage = new FlatFile(valid_filename, num_valid, 3, marius_options.storage.edges_dtype);
            test_edge_storage = new FlatFile(test_filename, num_test, 3, marius_options.storage.edges_dtype);
            break;
        }
        case BackendType::HostMemory: {
            train_edge_storage = new InMemory(train_filename, num_train, 3, marius_options.storage.edges_dtype, torch::kCPU);
            valid_edge_storage = new InMemory(valid_filename, num_valid, 3, marius_options.storage.edges_dtype, torch::kCPU);
            test_edge_storage = new InMemory(test_filename, num_test, 3, marius_options.storage.edges_dtype, torch::kCPU);
            break;
        }
        case BackendType::DeviceMemory: {
            train_edge_storage = new InMemory(train_filename, num_train, 3, marius_options.storage.edges_dtype, marius_options.general.device);
            valid_edge_storage = new InMemory(valid_filename, num_valid, 3, marius_options.storage.edges_dtype, marius_options.general.device);
            test_edge_storage = new InMemory(test_filename, num_test, 3, marius_options.storage.edges_dtype, marius_options.general.device);
            break;
        }
    }

    if (marius_options.storage.num_partitions > 1) {
        string train_edges_partitions = marius_options.path.experiment_directory
            + PathConstants::edges_directory
            + PathConstants::edges_train_directory
            + PathConstants::edge_partition_offsets_file;
//        string validation_edges_partitions = marius_options.path.experiment_directory
//                                             + PathConstants::edges_directory
//                                             + PathConstants::edges_validation_directory
//                                             + PathConstants::edge_partition_offsets_file;
//        string test_edges_partitions = marius_options.path.experiment_directory
//                                       + PathConstants::edges_directory
//                                       + PathConstants::edges_test_directory
//                                       + PathConstants::edge_partition_offsets_file;

        if (marius_options.storage.remove_preprocessed) {
            std::filesystem::rename(marius_options.path.train_edges_partitions, train_edges_partitions);
//            std::filesystem::rename(marius_options.path.validation_edges_partitions, validation_edges_partitions);
//            std::filesystem::rename(marius_options.path.test_edges_partitions, test_edges_partitions);
        } else {
            std::filesystem::copy_file(marius_options.path.train_edges_partitions, train_edges_partitions, std::filesystem::copy_options::update_existing);
//            std::filesystem::copy_file(marius_options.path.validation_edges_partitions, validation_edges_partitions);
//            std::filesystem::copy_file(marius_options.path.test_edges_partitions, test_edges_partitions);
        }
        train_edge_storage->readPartitionSizes(train_edges_partitions);
//        valid_edge_storage->readPartitionSizes(validation_edges_partitions);
//        test_edge_storage->readPartitionSizes(test_edges_partitions);
    }

    if (marius_options.storage.shuffle_input_edges) {
        train_edge_storage->shuffle();
        valid_edge_storage->shuffle();
        test_edge_storage->shuffle();
    }

    return forward_as_tuple(train_edge_storage, valid_edge_storage, test_edge_storage);
}

tuple<Storage *, Storage *> initializeNodeEmbeddings(bool train) {

    string node_embedding_filename = marius_options.path.experiment_directory
        + PathConstants::embeddings_directory
        + PathConstants::embeddings_file
        + PathConstants::file_ext;
    string optimizer_state_filename = marius_options.path.experiment_directory
        + PathConstants::embeddings_directory
        + PathConstants::state_file
        + PathConstants::file_ext;

    int64_t num_nodes = marius_options.general.num_nodes;
    bool reinitialize = marius_options.storage.reinitialize_embeddings;

    if (reinitialize && train) {
        FlatFile *init_node_embedding_storage = new FlatFile(node_embedding_filename, marius_options.storage.embeddings_dtype);
        FlatFile *init_optimizer_state_storage = new FlatFile(optimizer_state_filename, marius_options.storage.embeddings_dtype);

        int64_t curr_num_nodes = 0;
        int64_t offset = 0;

        // initialize 10 million nodes at a time
        while (offset < num_nodes) {
            if (num_nodes - offset < 1E7) {
                curr_num_nodes = num_nodes - offset;
            } else {
                curr_num_nodes = 1E7;
            }

            Embeddings weights;
            if (marius_options.model.initialization_distribution == InitializationDistribution::Uniform) {
                weights = (2 * torch::rand({curr_num_nodes, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype) - 1).mul_(marius_options.model.scale_factor);
            } else if (marius_options.model.initialization_distribution == InitializationDistribution::Normal) {
                weights = torch::randn({curr_num_nodes, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype).mul_(marius_options.model.scale_factor);
            }

            OptimizerState emb_state = torch::zeros_like(weights);
            init_node_embedding_storage->append(weights);
            init_optimizer_state_storage->append(emb_state);

            offset += curr_num_nodes;
        }

        delete init_node_embedding_storage;
        delete init_optimizer_state_storage;
    }

    Storage *node_embedding_storage;
    Storage *optimizer_state_storage;

    switch (marius_options.storage.embeddings) {
        case BackendType::RocksDB: {
            SPDLOG_ERROR("Currently Unsupported");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            node_embedding_storage = new PartitionBufferStorage(node_embedding_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.storage.buffer_capacity, true);
            if (train) {
                optimizer_state_storage = new PartitionBufferStorage(optimizer_state_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.storage.buffer_capacity, false);
            }
            break;
        }
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            exit(-1);
        }
        case BackendType::HostMemory: {
            node_embedding_storage = new InMemory(node_embedding_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
            if (train) {
                optimizer_state_storage = new InMemory(optimizer_state_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
            }
            break;
        }
        case BackendType::DeviceMemory: {
            node_embedding_storage = new InMemory(node_embedding_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
            if (train) {
                optimizer_state_storage = new InMemory(optimizer_state_filename, num_nodes, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
            }
            break;
        }
    }

    return forward_as_tuple(node_embedding_storage, optimizer_state_storage);
}

tuple<Storage *, Storage *, Storage *, Storage *> initializeRelationEmbeddings(bool train) {

    string src_relation_embedding_filename = marius_options.path.experiment_directory
        + PathConstants::relations_directory
        + PathConstants::src_relations_file
        + PathConstants::file_ext;
    string dst_relation_embedding_filename = marius_options.path.experiment_directory
        + PathConstants::relations_directory
        + PathConstants::dst_relations_file
        + PathConstants::file_ext;
    string src_optimizer_state_filename = marius_options.path.experiment_directory
        + PathConstants::relations_directory
        + PathConstants::src_state_file
        + PathConstants::file_ext;
    string dst_optimizer_state_filename = marius_options.path.experiment_directory
        + PathConstants::relations_directory
        + PathConstants::dst_state_file
        + PathConstants::file_ext;

    int64_t num_relations = marius_options.general.num_relations;

    bool reinitialize = marius_options.storage.reinitialize_embeddings;

    if (reinitialize && train) {
        Relations src_relations;
        Relations dst_relations;
        if (marius_options.model.relation_operator == RelationOperatorType::Translation) {
            src_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
            dst_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
        } else if (marius_options.model.relation_operator == RelationOperatorType::Hadamard) {
            src_relations = torch::ones({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
            dst_relations = torch::ones({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
        } else if (marius_options.model.relation_operator == RelationOperatorType::ComplexHadamard) {
            src_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
            src_relations.narrow(1, 0, (marius_options.model.embedding_size / 2) - 1).fill_(1);
            dst_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
            dst_relations.narrow(1, 0, (marius_options.model.embedding_size / 2) - 1).fill_(1);
        } else {
            src_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
            dst_relations = torch::zeros({num_relations, marius_options.model.embedding_size}, marius_options.storage.embeddings_dtype);
        }
        OptimizerState src_state = torch::zeros_like(src_relations);
        OptimizerState dst_state = torch::zeros_like(dst_relations);

        FlatFile *init_src_embedding_storage = new FlatFile(src_relation_embedding_filename, src_relations);
        FlatFile *init_src_state_storage = new FlatFile(src_optimizer_state_filename, src_state);
        FlatFile *init_dst_embedding_storage = new FlatFile(dst_relation_embedding_filename, dst_relations);
        FlatFile *init_dst_state_storage = new FlatFile(dst_optimizer_state_filename, dst_state);

        delete init_src_embedding_storage;
        delete init_src_state_storage;
        delete init_dst_embedding_storage;
        delete init_dst_state_storage;
    }

    Storage *src_relation_embedding_storage;
    Storage *src_optimizer_state_storage;
    Storage *dst_relation_embedding_storage;
    Storage *dst_optimizer_state_storage;

    switch (marius_options.storage.relations) {
        case BackendType::RocksDB: {
            SPDLOG_ERROR("Currently Unsupported");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            SPDLOG_ERROR("Backend type not available for relation embeddings.");
            exit(-1);
        }
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            exit(-1);
        }
        case BackendType::HostMemory: {
            src_relation_embedding_storage = new InMemory(src_relation_embedding_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
            dst_relation_embedding_storage = new InMemory(dst_relation_embedding_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
            if (train) {
                src_optimizer_state_storage = new InMemory(src_optimizer_state_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
                dst_optimizer_state_storage = new InMemory(dst_optimizer_state_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, torch::kCPU);
            }
            break;
        }
        case BackendType::DeviceMemory: {
            src_relation_embedding_storage = new InMemory(src_relation_embedding_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
            dst_relation_embedding_storage = new InMemory(dst_relation_embedding_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
            if (train) {
                src_optimizer_state_storage = new InMemory(src_optimizer_state_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
                dst_optimizer_state_storage = new InMemory(dst_optimizer_state_filename, num_relations, marius_options.model.embedding_size, marius_options.storage.embeddings_dtype, marius_options.general.device);
            }
            break;
        }
    }

    return forward_as_tuple(src_relation_embedding_storage, src_optimizer_state_storage, dst_relation_embedding_storage, dst_optimizer_state_storage);
}

tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> initializeTrain() {
    string data_directory = marius_options.path.experiment_directory;

    torch::manual_seed(marius_options.general.random_seed);

    initOutputDir(data_directory);

    int64_t num_train = marius_options.general.num_train;
    int64_t num_valid = marius_options.general.num_valid;
    int64_t num_test = marius_options.general.num_test;
    int64_t num_nodes = marius_options.general.num_nodes;
    int64_t num_relations = marius_options.general.num_relations;

    bool train = true;
    tuple<Storage *, Storage *, Storage *> edge_storages = initializeEdges(train);
    tuple<Storage *, Storage *> node_embedding_storages = initializeNodeEmbeddings(train);
    tuple<Storage *, Storage *, Storage *, Storage *> relation_embedding_storages = initializeRelationEmbeddings(train);

    return std::tuple_cat(edge_storages, node_embedding_storages, relation_embedding_storages);
}

tuple<Storage *, Storage *, Storage *, Storage *> initializeEval() {
    string data_directory = marius_options.path.experiment_directory;

    torch::manual_seed(marius_options.general.random_seed);

    initOutputDir(data_directory);

    int64_t num_train = marius_options.general.num_train;
    int64_t num_valid = marius_options.general.num_valid;
    int64_t num_test = marius_options.general.num_test;
    int64_t num_nodes = marius_options.general.num_nodes;
    int64_t num_relations = marius_options.general.num_relations;

    bool train = false;
    tuple<Storage *, Storage *, Storage *> edge_storages = initializeEdges(train);
    tuple<Storage *, Storage *> node_embedding_storages = initializeNodeEmbeddings(train);
    tuple<Storage *, Storage *, Storage *, Storage *> relation_embedding_storages = initializeRelationEmbeddings(train);

    return std::forward_as_tuple(std::get<2>(edge_storages), std::get<0>(node_embedding_storages), std::get<0>(relation_embedding_storages), std::get<2>(relation_embedding_storages));
}

void freeTrainStorage(Storage *train_edges,
                      Storage *eval_edges,
                      Storage *test_edges,
                      Storage *embeddings,
                      Storage *emb_state,
                      Storage *src_rel,
                      Storage *src_rel_state,
                      Storage *dst_rel,
                      Storage *dst_rel_state) {

    switch (marius_options.storage.edges) {
        case BackendType::RocksDB:
        case BackendType::PartitionBuffer: {
            SPDLOG_ERROR("Backend type not available for edges.");
            exit(-1);
        }
        case BackendType::FlatFile: {
            delete (FlatFile *) train_edges;
            delete (FlatFile *) eval_edges;
            delete (FlatFile *) test_edges;
            break;
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) train_edges;
            delete (InMemory *) eval_edges;
            delete (InMemory *) test_edges;
            break;
        }
    }

    switch (marius_options.storage.embeddings) {
        case BackendType::RocksDB:
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            delete (PartitionBufferStorage *) embeddings;
            delete (PartitionBufferStorage *) emb_state;
            break;
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) embeddings;
            delete (InMemory *) emb_state;
            break;
        }
    }

    switch (marius_options.storage.relations) {
        case BackendType::RocksDB:
        case BackendType::PartitionBuffer:
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for relation embeddings.");
            exit(-1);
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) src_rel;
            delete (InMemory *) src_rel_state;
            delete (InMemory *) dst_rel;
            delete (InMemory *) dst_rel_state;
            break;
        }
    }
}

void freeEvalStorage(Storage *test_edges, Storage *embeddings, Storage *src_rels, Storage *dst_rels) {
    switch (marius_options.storage.edges) {
        case BackendType::RocksDB: {
            SPDLOG_ERROR("Currently Unsupported");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            SPDLOG_ERROR("Backend type not available for edges.");
            exit(-1);
        }
        case BackendType::FlatFile: {
            delete (FlatFile *) test_edges;
            break;
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) test_edges;
            break;
        }
    }

    switch (marius_options.storage.embeddings) {
        case BackendType::RocksDB:
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for embeddings.");
            exit(-1);
        }
        case BackendType::PartitionBuffer: {
            delete (PartitionBufferStorage *) embeddings;
            break;
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) embeddings;
            break;
        }
    }

    switch (marius_options.storage.relations) {
        case BackendType::RocksDB:
        case BackendType::PartitionBuffer:
        case BackendType::FlatFile: {
            SPDLOG_ERROR("Backend type not available for relation embeddings.");
            exit(-1);
        }
        case BackendType::HostMemory:
        case BackendType::DeviceMemory: {
            delete (InMemory *) src_rels;
            delete (InMemory *) dst_rels;
            break;
        }
    }
}
