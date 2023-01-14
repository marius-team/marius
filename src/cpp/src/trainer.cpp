//
// Created by Jason Mohoney on 2/28/20.
//

#include "trainer.h"

#include "logger.h"

using std::tie;
using std::get;

PipelineTrainer::PipelineTrainer(DataLoader *dataloader, std::shared_ptr<Model>model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch) {
    dataloader_ = dataloader;
    learning_task_ = dataloader_->graph_storage_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = new ProgressReporter(item_name, num_items, logs_per_epoch);

    if (model->current_device_.is_cuda()) {
        pipeline_ = new PipelineGPU(dataloader, model, true, progress_reporter_, pipeline_config);
    } else {
        pipeline_ = new PipelineCPU(dataloader, model, true, progress_reporter_, pipeline_config);
    }
}

void PipelineTrainer::train(int num_epochs) {
    dataloader_->setTrainSet();
    dataloader_->loadStorage();
    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        pipeline_->start();
        pipeline_->waitComplete();
        pipeline_->stopAndFlush();
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float) num_items / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
    dataloader_->unloadStorage(true);
}

SynchronousTrainer::SynchronousTrainer(DataLoader *dataloader, std::shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->graph_storage_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = new ProgressReporter(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {
    dataloader_->setTrainSet();
    dataloader_->loadStorage();
    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double sample = 0.0;
        double load = 0.0;
        double transfer = 0.0;
        double compute = 0.0;
        double num = 0.0;

        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {

            // gets data and parameters for the next batch
            Batch *batch = dataloader_->getBatch();

            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                // transfers batch to the GPU
                batch->to(model_->current_device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }

            // compute forward and backward pass of the model
            model_->train_batch(batch);

            // transfer gradients and update parameters
            if (batch->unique_node_embeddings_.defined()) {
                batch->accumulateGradients(model_->model_config_->embeddings->optimizer->options->learning_rate);

                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddingsForBatch(batch, true);
                }

                dataloader_->updateEmbeddingsForBatch(batch, false);
            }

            batch->clear();

            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);

            sample += batch->sample_;
            load += batch->load_;
            transfer += batch->transfer_;
            compute += batch->compute_;
            num += 1;
        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        SPDLOG_INFO("Num batches: {}", num);
        SPDLOG_INFO("Sample avg: {}", sample/num);
        SPDLOG_INFO("Load avg: {}", load/num);
        SPDLOG_INFO("Transfer avg: {}", transfer/num);
        SPDLOG_INFO("Compute avg: {}", compute/num);

        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float) num_items / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
    dataloader_->unloadStorage(true);
}

SynchronousMultiGPUTrainer::SynchronousMultiGPUTrainer(DataLoader *dataloader, std::shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->graph_storage_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = new ProgressReporter(item_name, num_items, logs_per_epoch);
}

void SynchronousMultiGPUTrainer::train(int num_epochs) {

    dataloader_->setTrainSet();
    dataloader_->loadStorage();
    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {

            // get a batch and split into num_device sub batches. Splits are based on the edges in link prediction, and on the nodes in node classification
            std::vector<Batch *> sub_batches = dataloader_->getSubBatches();

            #pragma omp parallel for
            for (int i = 0; i < model_->devices_.size(); i++) {
                sub_batches[i]->to(model_->devices_[i]);
                dataloader_->loadGPUParameters(sub_batches[i]);
            }

            // The model is replicated on all devices
            model_->train_batch(sub_batches);

            #pragma omp parallel for
            for (int i = 0; i < model_->devices_.size(); i++) {
                // Aggregate node embedding gradients and optimizer state.
                sub_batches[i]->accumulateGradients(model_->model_config_->embeddings->optimizer->options->learning_rate);

                // synchronize on GPU or CPU?
                sub_batches[i]->embeddingsToHost();
            }

            // Merge the gradients for each subbatch into a single batch
            Batch *batch = new Batch(sub_batches);

            // Update node embeddings and optimizer state
            dataloader_->updateEmbeddingsForBatch(batch, false);

            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);

            delete batch;
        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);

        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();
        timer.stop();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float) num_items / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);
    }
    dataloader_->unloadStorage(true);
}

