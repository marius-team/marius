//
// Created by jasonmohoney on 10/4/19.
//

#ifndef MARIUS_IO_H
#define MARIUS_IO_H

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common/datatypes.h"
#include "storage/graph_storage.h"
#include "storage/storage.h"

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> initializeEdges(shared_ptr<StorageConfig> storage_config,
                                                                                                               LearningTask learning_task);

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>> initializeNodeEmbeddings(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config,
                                                                              bool reinitialize, bool train, std::shared_ptr<InitConfig> init_config);

std::tuple<shared_ptr<Storage>, shared_ptr<Storage>, shared_ptr<Storage>> initializeNodeIds(shared_ptr<StorageConfig> storage_config);

shared_ptr<Storage> initializeRelationFeatures(shared_ptr<StorageConfig> storage_config);

shared_ptr<Storage> initializeNodeFeatures(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

shared_ptr<Storage> initializeNodeLabels(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

shared_ptr<GraphModelStorage> initializeStorageLinkPrediction(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize,
                                                              bool train, std::shared_ptr<InitConfig> init_config);

shared_ptr<GraphModelStorage> initializeStorageNodeClassification(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize,
                                                                  bool train, std::shared_ptr<InitConfig> init_config);

shared_ptr<GraphModelStorage> initializeStorage(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config, bool reinitialize, bool train,
                                                std::shared_ptr<InitConfig> init_config = nullptr);

#endif  // MARIUS_IO_H
