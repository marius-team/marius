//
// Created by Jason Mohoney on 2/28/20.
//
#ifndef MARIUS_TRAINER_H
#define MARIUS_TRAINER_H

#include "dataset.h"
#include "pipeline.h"

class Trainer {
  public:
    DataSet *data_set_;

    virtual ~Trainer() { };
    virtual void train(int num_epochs = 1) = 0;
};

class PipelineTrainer : public Trainer {
    Pipeline *pipeline_;
  public:
    PipelineTrainer(DataSet *data_set, Model *model);

    void train(int num_epochs = 1) override;
};


class SynchronousTrainer : public Trainer  {
    Model *model_;
  public:
    SynchronousTrainer(DataSet *data_set, Model *model);

    void train(int num_epochs = 1) override;
};

#endif //MARIUS_TRAINER_H
