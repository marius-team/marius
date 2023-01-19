//
// Created by Jason Mohoney on 2/28/20.
//

#include "pipeline/evaluator.h"

#include "configuration/constants.h"
#include "reporting/logger.h"

PipelineEvaluator::PipelineEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config) {
    dataloader_ = dataloader;

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, false, nullptr, pipeline_config);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, false, nullptr, pipeline_config);
    }

    pipeline_->initialize();
}

void PipelineEvaluator::evaluate(bool validation) {
    if (!dataloader_->single_dataset_) {
        if (validation) {
            SPDLOG_INFO("Evaluating validation set");
            dataloader_->setValidationSet();
        } else {
            SPDLOG_INFO("Evaluating test set");
            dataloader_->setTestSet();
        }
    }

    dataloader_->initializeBatches(false);

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            dataloader_->graph_storage_->sortAllEdges();
        }
    }

    Timer timer = Timer(false);
    timer.start();
    pipeline_->start();
    pipeline_->waitComplete();
    pipeline_->pauseAndFlush();
    pipeline_->model_->reporter_->report();
    timer.stop();

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms", epoch_time);
}

SynchronousEvaluator::SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {
    if (!dataloader_->single_dataset_) {
        if (validation) {
            SPDLOG_INFO("Evaluating validation set");
            dataloader_->setValidationSet();
        } else {
            SPDLOG_INFO("Evaluating test set");
            dataloader_->setTestSet();
        }
    }

    dataloader_->initializeBatches(false);

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            dataloader_->graph_storage_->sortAllEdges();
        }
    }

    Timer timer = Timer(false);
    timer.start();
    int num_batches = 0;

    while (dataloader_->hasNextBatch()) {
        shared_ptr<Batch> batch = dataloader_->getBatch();
        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
            batch->to(model_->device_);
        }
        dataloader_->loadGPUParameters(batch);

        model_->evaluate_batch(batch);

        dataloader_->finishedBatch();
        batch->clear();
        num_batches++;
    }
    timer.stop();

    model_->reporter_->report();
}