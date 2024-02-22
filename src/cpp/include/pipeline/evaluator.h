//
// Created by Jason Mohoney on 2/28/20.
//

#ifndef MARIUS_EVALUATOR_H
#define MARIUS_EVALUATOR_H

#include <iostream>

#include "data/dataloader.h"
#include "pipeline_cpu.h"
#include "pipeline_gpu.h"

/**
  The evaluator runs the evaluation process using the given model and dataset.
*/
class Evaluator {
   public:
    shared_ptr<DataLoader> dataloader_;

    virtual ~Evaluator(){};

    /**
      Runs evaluation process.
      @param validation If true, evaluate on validation set. Otherwise evaluate on test set
    */
    virtual void evaluate(bool validation) = 0;
};

class PipelineEvaluator : public Evaluator {
    shared_ptr<Pipeline> pipeline_;

   public:
    PipelineEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config);

    void evaluate(bool validation) override;
};

class SynchronousEvaluator : public Evaluator {
    shared_ptr<Model> model_;

   public:
    SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model);

    void evaluate(bool validation) override;
};

#endif  // MARIUS_EVALUATOR_H
