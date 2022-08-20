//
// Created by Jason Mohoney on 2/28/20.
//

#include "evaluator.h"

#include "configuration/constants.h"
#include "logger.h"

PipelineEvaluator::PipelineEvaluator(DataLoader *dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config) {
    dataloader_ = dataloader;

    if (model->current_device_.is_cuda()) {
        pipeline_ = new PipelineGPU(dataloader, model, false, nullptr, pipeline_config);
    } else {
        pipeline_ = new PipelineCPU(dataloader, model, false, nullptr, pipeline_config);
    }

    pipeline_->initialize();
}

void PipelineEvaluator::evaluate(bool validation) {

    if (validation) {
        SPDLOG_INFO("Evaluating validation set");
        dataloader_->setValidationSet();
    } else {
        SPDLOG_INFO("Evaluating test set");
        dataloader_->setTestSet();
    }

    dataloader_->loadStorage();
    Timer timer = Timer(false);
    timer.start();
    pipeline_->start();
    pipeline_->waitComplete();
    pipeline_->stopAndFlush();
    pipeline_->model_->reporter_->report();
    timer.stop();

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms", epoch_time);
    dataloader_->unloadStorage();
}

SynchronousEvaluator::SynchronousEvaluator(DataLoader *dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {

    if (validation) {
        SPDLOG_INFO("Evaluating validation set");
        dataloader_->setValidationSet();
    } else {
        SPDLOG_INFO("Evaluating test set");
        dataloader_->setTestSet();
    }

    dataloader_->loadStorage();

    Timer timer = Timer(false);
    timer.start();
    int num_batches = 0;

    while (dataloader_->hasNextBatch()) {
        Batch *batch = dataloader_->getBatch();
        batch->to(model_->current_device_);
        dataloader_->loadGPUParameters(batch);
        model_->evaluate(batch, dataloader_->graph_storage_->filtered_eval_);
        dataloader_->finishedBatch();
        batch->clear();
        num_batches++;
    }
    timer.stop();

    model_->reporter_->report();

    dataloader_->unloadStorage();
}