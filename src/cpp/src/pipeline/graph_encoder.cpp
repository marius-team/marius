//
// Created by Jason Mohoney on 1/22/22.
//

#include "pipeline/graph_encoder.h"

#include "reporting/logger.h"

using std::get;
using std::tie;

PipelineGraphEncoder::PipelineGraphEncoder(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config,
                                           int logs_per_epoch) {
    dataloader_ = dataloader;

    std::string item_name = "Nodes";
    int64_t num_items = dataloader_->graph_storage_->getNumNodes();

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);

    if (model->device_.is_cuda()) {
        pipeline_ = std::make_shared<PipelineGPU>(dataloader, model, true, progress_reporter_, pipeline_config, true);
    } else {
        pipeline_ = std::make_shared<PipelineCPU>(dataloader, model, true, progress_reporter_, pipeline_config, true);
    }
}

void PipelineGraphEncoder::encode(bool separate_layers) {
    Timer timer = Timer(false);
    timer.start();

    pipeline_->start();
    pipeline_->waitComplete();
    pipeline_->pauseAndFlush();
    progress_reporter_->clear();

    timer.stop();

    std::string item_name = "Nodes";
    int64_t num_items = dataloader_->graph_storage_->getNumNodes();

    int64_t epoch_time = timer.getDuration();
    float items_per_second = (float)num_items / ((float)epoch_time / 1000);
    SPDLOG_INFO("Encode took: {}ms", epoch_time);
    SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
}

SynchronousGraphEncoder::SynchronousGraphEncoder(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;

    std::string item_name = "Nodes";
    int64_t num_items = dataloader_->graph_storage_->getNumNodes();

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousGraphEncoder::encode(bool separate_layers) {
    dataloader_->setEncode();
    Timer timer = Timer(false);
    timer.start();
    SPDLOG_INFO("Start full graph encode");

    while (dataloader_->hasNextBatch()) {
        shared_ptr<Batch> batch = dataloader_->getBatch();
        batch->to(model_->device_);
        dataloader_->loadGPUParameters(batch);

        torch::Tensor encoded_nodes = model_->encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
        batch->clear();

        encoded_nodes = encoded_nodes.contiguous().to(torch::kCPU);

        if (model_->device_.is_cuda()) {
            torch::cuda::synchronize();
        }

        dataloader_->graph_storage_->updatePutEncodedNodesRange(batch->start_idx_, batch->batch_size_, encoded_nodes);
        dataloader_->finishedBatch();
    }

    timer.stop();
    SPDLOG_INFO("Encode Complete: {}s", (double)timer.getDuration() / 1000);
}
