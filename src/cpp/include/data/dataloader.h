//
// Created by jasonmohoney on 10/4/19.
//

#ifndef MARIUS_DATASET_H
#define MARIUS_DATASET_H

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "batch.h"
#include "common/datatypes.h"
#include "configuration/config.h"
#include "data/samplers/edge.h"
#include "data/samplers/negative.h"
#include "data/samplers/neighbor.h"
#include "storage/graph_storage.h"
#include "storage/storage.h"

class DataLoader {
   public:
    bool train_;
    int epochs_processed_;
    int64_t batches_processed_;
    int64_t current_edge_;
    std::mutex *sampler_lock_;
    vector<shared_ptr<Batch>> batches_;
    int batch_size_;

    bool single_dataset_;

    int batch_id_offset_;
    vector<shared_ptr<Batch>>::iterator batch_iterator_;
    std::mutex *batch_lock_;
    std::condition_variable *batch_cv_;
    bool waiting_for_batches_;
    int batches_left_;
    int total_batches_processed_;
    bool all_read_;

    vector<torch::Tensor> buffer_states_;

    // Link prediction
    vector<torch::Tensor> edge_buckets_per_buffer_;
    vector<torch::Tensor>::iterator edge_buckets_per_buffer_iterator_;

    // Node classification
    vector<torch::Tensor> node_ids_per_buffer_;
    vector<torch::Tensor>::iterator node_ids_per_buffer_iterator_;

    shared_ptr<NeighborSampler> training_neighbor_sampler_;
    shared_ptr<NeighborSampler> evaluation_neighbor_sampler_;

    shared_ptr<NegativeSampler> training_negative_sampler_;
    shared_ptr<NegativeSampler> evaluation_negative_sampler_;

    Timestamp timestamp_;

    shared_ptr<GraphModelStorage> graph_storage_;

    shared_ptr<EdgeSampler> edge_sampler_;
    shared_ptr<NegativeSampler> negative_sampler_;
    shared_ptr<NeighborSampler> neighbor_sampler_;

    shared_ptr<TrainingConfig> training_config_;
    shared_ptr<EvaluationConfig> evaluation_config_;
    bool only_root_features_;

    LearningTask learning_task_;

    DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
               shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config);

    DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler = nullptr,
               shared_ptr<NeighborSampler> neighbor_sampler = nullptr, bool train = false);

    ~DataLoader();

    void setBufferOrdering();

    void setActiveEdges();

    void setActiveNodes();

    void initializeBatches(bool prepare_encode = false);

    void clearBatches();

    /**
     * Check to see whether another batch exists.
     * @return True if batch exists, false if not
     */
    bool hasNextBatch();

    shared_ptr<Batch> getNextBatch();

    /**
     * Notify that the batch has been completed. Used for concurrency control.
     */
    void finishedBatch();

    /**
     * Gets the next batch to be processed by the pipeline.
     * Loads edges from storage
     * Constructs negative negative edges
     * Loads CPU embedding parameters
     * @return The next batch
     */
    shared_ptr<Batch> getBatch(at::optional<torch::Device> device = c10::nullopt, bool perform_map = false);

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load edges into.
     */
    void edgeSample(shared_ptr<Batch> batch);

    /**
     * Creates a mapping from global node ids into batch local node ids
     * @param batch: Batch to map
     */
    void mapEdges(shared_ptr<Batch> batch, bool use_negs, bool use_nbrs, bool set_map);

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load nodes into.
     */
    void nodeSample(shared_ptr<Batch> batch);

    /**
     * Samples negatives for the batch using the dataloader's negative sampler
     * @param batch: Batch object to load negative samples into.
     */
    void negativeSample(shared_ptr<Batch> batch);

    /**
     * Loads CPU parameters into batch
     * @param batch: Batch object to load parameters into.
     */
    void loadCPUParameters(shared_ptr<Batch> batch);

    /**
     * Loads GPU parameters into batch
     * @param batch Batch object to load parameters into.
     */
    void loadGPUParameters(shared_ptr<Batch> batch);

    /**
     * Applies gradient updates to underlying storage
     * @param batch: Batch object to apply updates from.
     * @param gpu: If true, only the gpu parameters will be updated.
     */
    void updateEmbeddings(shared_ptr<Batch> batch, bool gpu);

    /**
     * Notify that the epoch has been completed. Prepares dataset for a new epoch.
     */
    void nextEpoch();

    /**
     * Load graph from storage.
     */
    void loadStorage();

    bool epochComplete() { return (batches_left_ == 0) && all_read_; }

    /**
     * Unload graph from storage.
     * @param write Set to true to write embedding table state to disk
     */
    void unloadStorage(bool write = false) { graph_storage_->unload(write); }

    /**
     * Gets the number of edges from the graph storage.
     * @return Number of edges in the graph
     */
    int64_t getNumEdges() { return graph_storage_->getNumEdges(); }

    int64_t getEpochsProcessed() { return epochs_processed_; }

    int64_t getBatchesProcessed() { return batches_processed_; }

    bool isTrain() { return train_; }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to training set.
     */
    void setTrainSet() {
        if (single_dataset_) {
            throw MariusRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = training_config_->batch_size;
            train_ = true;
            graph_storage_->setTrainSet();
            negative_sampler_ = training_negative_sampler_;
            neighbor_sampler_ = training_neighbor_sampler_;
            loadStorage();
        }
    }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to validation set.
     */
    void setValidationSet() {
        if (single_dataset_) {
            throw MariusRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setValidationSet();
            negative_sampler_ = evaluation_negative_sampler_;
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            loadStorage();
        }
    }

    void setTestSet() {
        if (single_dataset_) {
            throw MariusRuntimeException("This dataloader only has a single dataset and cannot switch");
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setTestSet();
            negative_sampler_ = evaluation_negative_sampler_;
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            loadStorage();
        }
    }

    void setEncode() {
        if (single_dataset_) {
            loadStorage();
            initializeBatches(true);
        } else {
            batch_size_ = evaluation_config_->batch_size;
            train_ = false;
            graph_storage_->setTrainSet();
            neighbor_sampler_ = evaluation_neighbor_sampler_;
            loadStorage();
            initializeBatches(true);
        }
    }
};

#endif  // MARIUS_DATASET_H
