//
// Created by jasonmohoney on 10/4/19.
//
#ifndef MARIUS_IO_H
#define MARIUS_IO_H

#include <dataset.h>
#include <datatypes.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <storage.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <torch/script.h>
#include <util.h>

using std::vector;
using std::string;
using std::tuple;

void createDir(const string &path);

void initOutputDir(const string &output_directory);

tuple<EdgeList, EdgeList, EdgeList, int64_t, int64_t> prepareEdges();

tuple<Storage *, Storage *, Storage *> initializeEdges(bool train);

tuple<Storage *, Storage *> initializeNodeEmbeddings(bool train);

tuple<Storage *, Storage *, Storage *, Storage *> initializeRelationEmbeddings(bool train);

tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> initializeTrain();

tuple<Storage *, Storage *, Storage *, Storage *> initializeEval();

void freeTrainStorage(Storage *train_edges,
                      Storage *eval_edges,
                      Storage *test_edges,
                      Storage *embeddings,
                      Storage *emb_state,
                      Storage *src_rel,
                      Storage *src_rel_state,
                      Storage *dst_rel,
                      Storage *dst_rel_state);

void freeEvalStorage(Storage *test_edges, Storage *embeddings, Storage *src_rels, Storage *dst_rels);

#endif //MARIUS_IO_H
