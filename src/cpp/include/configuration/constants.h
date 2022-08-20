//
// Created by Jason Mohoney on 2/18/20.
//

#ifndef MARIUS_CONSTANTS_H
#define MARIUS_CONSTANTS_H

#include <string>

#include "datatypes.h"

#define MISSING_STR "???"

namespace PathConstants {
    const string model_file = "model.pt";
    const string model_state_file = "model_state.pt";
    const string training = "train_";
    const string validation = "validation_";
    const string test = "test_";
    const string edges_directory = "edges/";
    const string edges_file = "edges";
    const string edge_partition_offsets_file = "partition_offsets.txt";
    const string nodes_directory = "nodes/";
    const string nodes_file = "nodes";
    const string features_file = "features";
    const string labels_file = "labels";
    const string embeddings_file = "embeddings";
    const string embeddings_state_file = "embeddings_state";
    const string file_ext = ".bin";
};

#endif //MARIUS_CONSTANTS_H
