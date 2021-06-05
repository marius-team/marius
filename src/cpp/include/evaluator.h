//
// Created by Jason Mohoney on 2/28/20.
//

#ifndef MARIUS_EVALUATOR_H
#define MARIUS_EVALUATOR_H

#include <iostream>

#include "dataset.h"
#include "pipeline.h"

class Evaluator {
  public:
    DataSet *data_set_;
    virtual ~Evaluator() { };
    virtual void evaluate(bool validation) = 0;
};

class PipelineEvaluator : public Evaluator {
    Pipeline *pipeline_;
  public:
    PipelineEvaluator(DataSet *data_set, Model *model);

    void evaluate(bool validation) override;
};

class SynchronousEvaluator : public Evaluator {
    Model *model_;
  public:
    SynchronousEvaluator(DataSet *data_set, Model *model);

    void evaluate(bool validation) override;
};

#endif //MARIUS_EVALUATOR_H

