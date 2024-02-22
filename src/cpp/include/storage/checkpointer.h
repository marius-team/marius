//
// Created by Jason Mohoney on 12/15/21.
//

#ifndef MARIUS_CHECKPOINTER_H
#define MARIUS_CHECKPOINTER_H

#include "data/dataloader.h"
#include "nn/model.h"
#include "storage/storage.h"

struct CheckpointMeta {
    string name = "checkpoint";
    int num_epochs = -1;
    int checkpoint_id = -1;

    bool link_prediction = true;
    bool has_state = false;
    bool has_encoded = false;
    bool has_model = true;
};

class Checkpointer {
   public:
    std::shared_ptr<Model> model_;
    shared_ptr<GraphModelStorage> storage_;
    std::shared_ptr<CheckpointConfig> config_;

    Checkpointer(std::shared_ptr<Model> model, shared_ptr<GraphModelStorage> storage, std::shared_ptr<CheckpointConfig> config);

    Checkpointer(){};

    void saveMetadata(string directory, CheckpointMeta checkpoint_meta);

    CheckpointMeta loadMetadata(string directory);

    std::tuple<std::shared_ptr<Model>, shared_ptr<GraphModelStorage>, CheckpointMeta> load(string checkpoint_dir, std::shared_ptr<MariusConfig> marius_config,
                                                                                           bool train);

    void save(string checkpoint_dir, CheckpointMeta checkpoint_meta);

    void create_checkpoint(string checkpoint_dir, CheckpointMeta checkpoint_meta, int epochs);
};

#endif  // MARIUS_CHECKPOINTER_H