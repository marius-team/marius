//
// Created by Jason Mohoney on 2/28/20.
//

#include "evaluator.h"

#include "config.h"
#include "logger.h"

PipelineEvaluator::PipelineEvaluator(DataSet *data_set, Model *model) {
    data_set_ = data_set;

    timespec sleep_time{};
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 100 * MILLISECOND; // report progress every 100 ms

    if (marius_options.general.device == torch::kCUDA) {
        pipeline_ = new PipelineGPU(data_set_, model, false, sleep_time);
    } else {
        pipeline_ = new PipelineCPU(data_set_, model, false, sleep_time);
    }

    pipeline_->initialize();
}

void PipelineEvaluator::evaluate(bool validation) {

    if (validation) {
        data_set_->setValidationSet();
    } else {
        data_set_->setTestSet();
    }

    data_set_->loadStorage();
    Timer timer = Timer(false);
    timer.start();
    SPDLOG_INFO("Starting evaluating");
    data_set_->syncEmbeddings();
    pipeline_->start();
    pipeline_->waitComplete();
    pipeline_->stopAndFlush();
    pipeline_->reportMRR();
    data_set_->nextEpoch();
    timer.stop();

    int64_t epoch_time = timer.getDuration();
    SPDLOG_INFO("Evaluation complete: {}ms", epoch_time);
    data_set_->unloadStorage();
}

SynchronousEvaluator::SynchronousEvaluator(DataSet *data_set, Model *model) {
    data_set_ = data_set;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {

    if (validation) {
        data_set_->setValidationSet();
    } else {
        data_set_->setTestSet();
    }

    data_set_->loadStorage();
    Timer timer = Timer(false);
    timer.start();
    int num_batches = 0;
    while (data_set_->hasNextBatch()) {

        Batch *batch = data_set_->getBatch(); // gets the node embeddings and edges for the batch

        batch->embeddingsToDevice(0); // transfers the node embeddings to the GPU

        data_set_->loadGPUParameters(batch); // load the edge-type embeddings to batch

        batch->prepareBatch();

        model_->evaluate(batch);

        num_batches++;
    }
    timer.stop();

    auto ranks = data_set_->accumulateRanks();
    double avg_ranks = ranks.mean().item<double>();
    double mrr = ranks.reciprocal().mean().item<double>();
    double auc = data_set_->accumulateAuc();
    double ranks1 = (double) ranks.le(1).nonzero().size(0) / ranks.size(0);
    double ranks5 = (double) ranks.le(5).nonzero().size(0) / ranks.size(0);
    double ranks10 = (double) ranks.le(10).nonzero().size(0) / ranks.size(0);
    double ranks20 = (double) ranks.le(20).nonzero().size(0) / ranks.size(0);
    double ranks50 = (double) ranks.le(50).nonzero().size(0) / ranks.size(0);
    double ranks100 = (double) ranks.le(100).nonzero().size(0) / ranks.size(0);

    SPDLOG_INFO("Num Eval Edges: {}", data_set_->getNumEdges());
    SPDLOG_INFO("Num Eval Batches: {}", data_set_->batches_processed_);
    SPDLOG_INFO("Auc: {:.3f}, Avg Ranks: {:.3f}, MRR: {:.3f}, Hits@1: {:.3f}, Hits@5: {:.3f}, Hits@10: {:.3f}, Hits@20: {:.3f}, Hits@50: {:.3f}, Hits@100: {:.3f}", auc, avg_ranks, mrr, ranks1, ranks5, ranks10,
                ranks20, ranks50, ranks100);


    data_set_->unloadStorage();
}
