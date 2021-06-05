//
// Created by Jason Mohoney on 2/28/20.
//

#include "trainer.h"

#include "logger.h"

using std::cout;
using std::tie;
using std::get;

PipelineTrainer::PipelineTrainer(DataSet *data_set, Model *model) {
    data_set_ = data_set;

    timespec sleep_time{};
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 100 * MILLISECOND; // report progress every 100 ms

    if (marius_options.general.device == torch::kCUDA) {
        pipeline_ = new PipelineGPU(data_set_, model, true, sleep_time);
    } else {
        pipeline_ = new PipelineCPU(data_set_, model, true, sleep_time);
    }

    pipeline_->initialize();
}

void PipelineTrainer::train(int num_epochs) {
    data_set_->loadStorage();
    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", data_set_->getEpochsProcessed() + 1);
        pipeline_->start();
        pipeline_->waitComplete();
        SPDLOG_INFO("################ Finished training epoch {} ################", data_set_->getEpochsProcessed() + 1);
        timer.stop();

        int64_t epoch_time = timer.getDuration();
        float edges_per_second = (float) data_set_->getNumEdges() / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime (Before shuffle/sync): {}ms", epoch_time);
        SPDLOG_INFO("Edges per Second (Before shuffle/sync): {}", edges_per_second);

        pipeline_->stopAndFlush();
        data_set_->nextEpoch();
        timer.stop();

        epoch_time = timer.getDuration();
        edges_per_second = (float) data_set_->getNumEdges() / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime (Including shuffle/sync): {}ms", epoch_time);
        SPDLOG_INFO("Edges per Second (Including shuffle/sync): {}", edges_per_second);
    }
    data_set_->unloadStorage();
}

SynchronousTrainer::SynchronousTrainer(DataSet *data_set, Model *model) {
    data_set_ = data_set;
    model_ = model;
}

void SynchronousTrainer::train(int num_epochs) {
    int64_t edges_processed = 0;
    data_set_->loadStorage();
    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", data_set_->getEpochsProcessed() + 1);
        int num_batches = 0;
        while (data_set_->hasNextBatch()) {

            Batch *batch = data_set_->getBatch(); // gets the node embeddings and edges for the batch

            batch->embeddingsToDevice(0); // transfers the node embeddings to the GPU

            data_set_->loadGPUParameters(batch); // load the edge-type embeddings to batch

            batch->prepareBatch();

            model_->train(batch);

            batch->accumulateGradients();

            batch->embeddingsToHost();

            data_set_->updateEmbeddingsForBatch(batch, true);
            data_set_->updateEmbeddingsForBatch(batch, false);

            edges_processed += batch->batch_size_;

            // report output to log
            int64_t num_batches_per_log = data_set_->getNumBatches() / marius_options.reporting.logs_per_epoch;
            if ((batch->batch_id_ + 1) % num_batches_per_log == 0) {
                SPDLOG_INFO("Total Edges Processed: {}, Percent Complete: {:.3f}", edges_processed, (double) (batch->batch_id_ + 1) / data_set_->getNumBatches());
            }

            num_batches++;
        }
        SPDLOG_INFO("################ Finished training epoch {} ################", data_set_->getEpochsProcessed() + 1);
        timer.stop();

        int64_t epoch_time = timer.getDuration();
        float edges_per_second = (float) data_set_->getNumEdges() / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime (Before shuffle/sync): {}ms", epoch_time);
        SPDLOG_INFO("Edges per Second (Before shuffle/sync): {}", edges_per_second);

        bool last_epoch = epoch < num_epochs - 1;
        data_set_->nextEpoch();
        timer.stop();

        epoch_time = timer.getDuration();
        edges_per_second = (float) data_set_->getNumEdges() / ((float) epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime (Including shuffle/sync): {}ms", epoch_time);
        SPDLOG_INFO("Edges per Second (Including shuffle/sync): {}", edges_per_second);
    }
    data_set_->unloadStorage();
}
