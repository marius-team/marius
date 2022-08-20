//
// Created by jasonmohoney on 10/4/19.
//

#ifndef MARIUS_IO_H
#define MARIUS_IO_H

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include "datatypes.h"
#include "graph_storage.h"
#include "storage.h"

#define MAX_NODE_EMBEDDING_INIT_SIZE 1E7 // how many node embeddings to initialize at one time

std::tuple<Storage *, Storage *, Storage *, Storage *> initializeEdges(shared_ptr<StorageConfig> storage_config, LearningTask learning_task);

std::tuple<Storage *, Storage *, Storage *> initializeEdgeFeatures(shared_ptr<StorageConfig> storage_config);

std::tuple<Storage *, Storage *> initializeNodeEmbeddings(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

std::tuple<Storage *, Storage *, Storage *> initializeNodeIds(shared_ptr<StorageConfig> storage_config);

Storage *initializeNodeFeatures(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

Storage *initializeNodeLabels(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

GraphModelStorage *initializeStorageLinkPrediction(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

GraphModelStorage *initializeStorageNodeClassification(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

GraphModelStorage *initializeStorage(std::shared_ptr<Model> model, shared_ptr<StorageConfig> storage_config);

GraphModelStorage *initializeTrainEdgesStorage(shared_ptr<StorageConfig> storage_config);

#endif //MARIUS_IO_H
