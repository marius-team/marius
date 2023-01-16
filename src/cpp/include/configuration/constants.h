//
// Created by Jason Mohoney on 2/18/20.
//

#ifndef MARIUS_CONSTANTS_H
#define MARIUS_CONSTANTS_H

#include <string>

#include "common/datatypes.h"

#define MISSING_STR "???"

#define MAX_NODE_EMBEDDING_INIT_SIZE 1E7  // how many node embeddings to initialize at one time

namespace PathConstants {
const string model_file = "model.pt";
const string model_state_file = "model_state.pt";
const string model_config_file = "model_config.yaml";

const string training = "train_";
const string validation = "validation_";
const string test = "test_";

const string dst_sort = "_dst_sort";

const string edges_directory = "edges/";
const string edges_file = "edges";
const string edge_partition_offsets_file = "partition_offsets.txt";

const string node_mapping_file = "node_mapping.txt";
const string relation_mapping_file = "relation_mapping.txt";

const string nodes_directory = "nodes/";
const string nodes_file = "nodes";
const string features_file = "features";
const string labels_file = "labels";
const string embeddings_file = "embeddings";
const string encoded_nodes_file = "encoded_nodes";
const string embeddings_state_file = "embeddings_state";

const string file_ext = ".bin";
const string checkpoint_metadata_file = "metadata.csv";
const string config_file = "config.yaml";

const string output_metrics_file = "metrics.txt";
const string output_scores_file = "scores.csv";
const string output_labels_file = "labels.csv";
};  // namespace PathConstants

#endif  // MARIUS_CONSTANTS_H
