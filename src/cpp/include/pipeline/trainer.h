//
// Created by Jason Mohoney on 2/28/20.
//
#ifndef MARIUS_TRAINER_H
#define MARIUS_TRAINER_H

#include "data/dataloader.h"
#include "pipeline_cpu.h"
#include "pipeline_gpu.h"

/**
  The trainer runs the training process using the given model for the specified number of epochs.
*/
class Trainer {
   public:
    shared_ptr<DataLoader> dataloader_;
    shared_ptr<ProgressReporter> progress_reporter_;
    LearningTask learning_task_;

    virtual ~Trainer(){};
    /**
      Runs training process for embeddings for specified number of epochs.
      @param num_epochs The number of epochs to train for
    */
    virtual void train(int num_epochs = 1) = 0;
};

class PipelineTrainer : public Trainer {
    shared_ptr<Pipeline> pipeline_;

   public:
    PipelineTrainer(shared_ptr<DataLoader> dataloader, std::shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch = 10);

    void train(int num_epochs = 1) override;
};

class SynchronousTrainer : public Trainer {
    std::shared_ptr<Model> model_;

   public:
    SynchronousTrainer(shared_ptr<DataLoader> dataloader, std::shared_ptr<Model> model, int logs_per_epoch = 10);

    void train(int num_epochs = 1) override;
};

#endif  // MARIUS_TRAINER_H
