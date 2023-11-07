//
// Created by Jason Mohoney on 2/28/20.
//

#include "pipeline/evaluator.h"

#include "configuration/constants.h"
#include "reporting/logger.h"

PipelineEvaluator::PipelineEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config,
                                     bool batch_worker, bool compute_worker, bool batch_worker_needs_remote, bool compute_worker_needs_remote) {
    dataloader_ = dataloader;

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, false, nullptr, pipeline_config, false,
                                                  batch_worker, compute_worker, batch_worker_needs_remote, compute_worker_needs_remote);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, false, nullptr, pipeline_config);
    }

//    pipeline_->initialize();
}

void PipelineEvaluator::evaluate(bool validation) {
    if (!dataloader_->single_dataset_ and dataloader_->batch_worker_) {
        if (validation) {
            SPDLOG_INFO("Evaluating validation set");
            dataloader_->setValidationSet();
        } else {
            SPDLOG_INFO("Evaluating test set");
            dataloader_->setTestSet();
        }
    } else {
        dataloader_->train_ = false;
    }

    if (dataloader_->batch_worker_) {
        dataloader_->initializeBatches(false);
    }

    if (dataloader_->evaluation_negative_sampler_ != nullptr and dataloader_->batch_worker_) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            dataloader_->graph_storage_->sortAllEdges();
        }
    }

    pipeline_->model_->distPrepareForTraining(true);

    Timer timer = Timer(false);
    timer.start();
    pipeline_->start();
    pipeline_->waitComplete();
    pipeline_->pauseAndFlush();
    timer.stop();

    pipeline_->model_->distNotifyCompleteAndWait(true);

    if (dataloader_->batch_worker_)
        pipeline_->model_->reporter_->report(validation);

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms", epoch_time);
}

SynchronousEvaluator::SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {
    //TODO: evaluate on batch construction worker 0 only?
    // treat similar to train, if a batch construction worker isn't doing eval, then it can just signal to it's
    // gpus that it's done and then wait for eval to finish

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

    model_->distPrepareForTraining();

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

    model_->distNotifyCompleteAndWait(true);

    model_->reporter_->report(validation);
}