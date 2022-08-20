//
// Created by Jason Mohoney on 2/28/20.
//
#ifndef MARIUS_TRAINER_H
#define MARIUS_TRAINER_H

#include "dataloader.h"
#include "pipeline.h"

/**
  The trainer runs the training process using the given model for the specified number of epochs.
*/
class Trainer {
  public:
    DataLoader *dataloader_;
    ProgressReporter *progress_reporter_;
    LearningTask learning_task_;

    virtual ~Trainer() { };
    /**
      Runs training process for embeddings for specified number of epochs.
      @param num_epochs The number of epochs to train for
    */
    virtual void train(int num_epochs = 1) = 0;
};

class PipelineTrainer : public Trainer {
    Pipeline *pipeline_;
  public:
    PipelineTrainer(DataLoader *sampler, std::shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch=10);

    void train(int num_epochs = 1) override;
};

class SynchronousTrainer : public Trainer  {
    std::shared_ptr<Model> model_;
  public:
    SynchronousTrainer(DataLoader *sampler, std::shared_ptr<Model> model, int logs_per_epoch=10);

    void train(int num_epochs = 1) override;
};

class SynchronousMultiGPUTrainer : public Trainer  {
    std::shared_ptr<Model> model_;
  public:
    SynchronousMultiGPUTrainer(DataLoader *sampler, std::shared_ptr<Model> model, int logs_per_epoch=10);

    void train(int num_epochs = 1) override;
};


#endif //MARIUS_TRAINER_H
