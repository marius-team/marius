//
// Created by Jason Mohoney on 12/15/21.
//

#include "storage/checkpointer.h"

#include <stdio.h>

#include <filesystem>

#include "configuration/util.h"
#include "reporting/logger.h"
#include "storage/io.h"

Checkpointer::Checkpointer(std::shared_ptr<Model> model, shared_ptr<GraphModelStorage> storage, std::shared_ptr<CheckpointConfig> config) {
    model_ = model;
    storage_ = storage;
    config_ = config;
}

void Checkpointer::create_checkpoint(string checkpoint_dir, CheckpointMeta checkpoint_meta, int epochs) {
    string tmp_checkpoint_dir = checkpoint_dir + "checkpoint_" + std::to_string(epochs) + "_tmp/";
    std::filesystem::create_directory(tmp_checkpoint_dir);

    std::string new_embeddings_file = tmp_checkpoint_dir + PathConstants::embeddings_file + PathConstants::file_ext;
    std::string new_embeddings_state_file = tmp_checkpoint_dir + PathConstants::embeddings_state_file + PathConstants::file_ext;

    std::string embeddings_file = checkpoint_dir + PathConstants::embeddings_file + PathConstants::file_ext;
    std::string embeddings_state_file = checkpoint_dir + PathConstants::embeddings_state_file + PathConstants::file_ext;

    if (std::filesystem::exists(embeddings_file)) {
        if (this->config_->save_encoded) std::filesystem::copy(embeddings_file, new_embeddings_file);
        if (this->config_->save_state) std::filesystem::copy(embeddings_state_file, new_embeddings_state_file);
    }

    this->save(tmp_checkpoint_dir, checkpoint_meta);

    string final_checkpoint_dir = checkpoint_dir + "checkpoint_" + std::to_string(epochs) + "/";
    std::filesystem::rename(tmp_checkpoint_dir, final_checkpoint_dir);

    int delete_checkpoint = epochs - this->config_->save_prev_num * this->config_->interval;
    if (delete_checkpoint > 0) {
        std::string delete_checkpoint_dir = checkpoint_dir + "checkpoint_" + std::to_string(delete_checkpoint) + "/";
        std::filesystem::remove_all(delete_checkpoint_dir);
    }
}

void Checkpointer::save(string checkpoint_dir, CheckpointMeta checkpoint_meta) {
    if (checkpoint_meta.has_model) {
        if (storage_->storage_ptrs_.node_embeddings != nullptr) {
            storage_->storage_ptrs_.node_embeddings->write();
        }
        model_->save(checkpoint_dir);
    }

    if (checkpoint_meta.has_state) {
        if (storage_->storage_ptrs_.node_optimizer_state != nullptr) {
            storage_->storage_ptrs_.node_optimizer_state->write();
        }
    }

    saveMetadata(checkpoint_dir, checkpoint_meta);

    //    if (storage_->base_directory_ != checkpoint_dir) {
    //
    //        createDir(checkpoint_dir, false);
    //        createDir(checkpoint_dir + PathConstants::edges_directory, false);
    //        createDir(checkpoint_dir + PathConstants::nodes_directory, false);
    //
    //        string node_mapping_filename = storage_->base_directory_
    //                                       + PathConstants::nodes_directory
    //                                       + PathConstants::node_mapping_file;
    //
    //        string output_node_mapping_filename = checkpoint_dir
    //                                              + PathConstants::nodes_directory
    //                                              + PathConstants::node_mapping_file;
    //
    //        if (fileExists(node_mapping_filename)) {
    //            copyFile(node_mapping_filename, output_node_mapping_filename);
    //        }
    //
    //        string relation_mapping_filename = storage_->base_directory_
    //                                           + PathConstants::edges_directory
    //                                           + PathConstants::relation_mapping_file;
    //
    //        string output_relation_mapping_filename = checkpoint_dir
    //                                                  + PathConstants::edges_directory
    //                                                  + PathConstants::relation_mapping_file;
    //
    //        if (fileExists(relation_mapping_filename)) {
    //            copyFile(relation_mapping_filename, output_relation_mapping_filename);
    //        }
    //
    //        // get node embeddings and model
    //        if (checkpoint_meta.has_model) {
    //            string node_embedding_filename = storage_->base_directory_
    //                                             + PathConstants::nodes_directory
    //                                             + PathConstants::embeddings_file
    //                                             + PathConstants::file_ext;
    //
    //            string output_node_embedding_filename = checkpoint_dir
    //                                                    + PathConstants::nodes_directory
    //                                                    + PathConstants::embeddings_file
    //                                                    + PathConstants::file_ext;
    //
    //
    //            if (fileExists(node_embedding_filename)) {
    //                copyFile(node_embedding_filename, output_node_embedding_filename);
    //            }
    //
    //            string model_filename = storage_->base_directory_ + PathConstants::model_file;
    //            string output_model_filename = checkpoint_dir + PathConstants::model_file;
    //
    //            if (fileExists(model_filename)) {
    //                copyFile(model_filename, output_model_filename);
    //            }
    //
    //            if (checkpoint_meta.has_state) {
    //                string node_state_filename = storage_->base_directory_
    //                                             + PathConstants::nodes_directory
    //                                             + PathConstants::embeddings_state_file
    //                                             + PathConstants::file_ext;
    //
    //                string output_node_state_filename = checkpoint_dir
    //                                                    + PathConstants::nodes_directory
    //                                                    + PathConstants::embeddings_state_file
    //                                                    + PathConstants::file_ext;
    //
    //                if (fileExists(node_state_filename)) {
    //                    copyFile(node_state_filename, output_node_state_filename);
    //                }
    //            }
    //        }
    //        copyFile(storage_->base_directory_ + PathConstants::config_file, checkpoint_dir + PathConstants::config_file);
    //
    //        copyFile(storage_->base_directory_ + PathConstants::checkpoint_metadata_file, checkpoint_dir + PathConstants::checkpoint_metadata_file);
    //    }
}

std::tuple<std::shared_ptr<Model>, shared_ptr<GraphModelStorage>, CheckpointMeta> Checkpointer::load(string checkpoint_dir,
                                                                                                     std::shared_ptr<MariusConfig> marius_config,
                                                                                                     bool train) {
    CheckpointMeta checkpoint_meta = loadMetadata(checkpoint_dir);

    std::vector<torch::Device> devices = devices_from_config(marius_config->storage);
    std::shared_ptr<Model> model = initModelFromConfig(marius_config->model,
                                                       devices,
                                                       marius_config->storage->dataset->num_relations,
                                                       train);
    model->load(checkpoint_dir, train);

    if (checkpoint_meta.link_prediction) {
        model->learning_task_ = LearningTask::LINK_PREDICTION;
    } else {
        model->learning_task_ = LearningTask::NODE_CLASSIFICATION;
    }

    shared_ptr<GraphModelStorage> storage = initializeStorage(model,
                                                              marius_config->storage,
                                                              false,
                                                              train);

    return std::forward_as_tuple(model, storage, checkpoint_meta);
}

CheckpointMeta Checkpointer::loadMetadata(string directory) {
    CheckpointMeta ret_meta;

    std::ifstream input_file;
    input_file.open(directory + PathConstants::checkpoint_metadata_file);

    std::string line;
    std::getline(input_file, line);
    ret_meta.name = line;

    std::getline(input_file, line);
    ret_meta.num_epochs = std::stoi(line);

    std::getline(input_file, line);
    ret_meta.checkpoint_id = std::stoi(line);

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.link_prediction;

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.has_state;

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.has_encoded;

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.has_model;

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.has_edges;

    std::getline(input_file, line);
    std::istringstream(line) >> ret_meta.has_features;

    return ret_meta;
}

void Checkpointer::saveMetadata(string directory, CheckpointMeta checkpoint_meta) {
    std::ofstream output_file;
    output_file.open(directory + PathConstants::checkpoint_metadata_file);

    output_file << checkpoint_meta.name << "\n";
    output_file << checkpoint_meta.num_epochs << "\n";
    output_file << checkpoint_meta.checkpoint_id << "\n";
    output_file << checkpoint_meta.link_prediction << "\n";
    output_file << checkpoint_meta.has_state << "\n";
    output_file << checkpoint_meta.has_encoded << "\n";
    output_file << checkpoint_meta.has_model << "\n";
    output_file << checkpoint_meta.has_edges << "\n";
    output_file << checkpoint_meta.has_features << "\n";
}