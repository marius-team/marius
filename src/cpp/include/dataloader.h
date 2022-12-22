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
#include "configuration/config.h"
#include "datatypes.h"
#include "graph_samplers.h"
#include "graph_storage.h"
#include "storage.h"

/**
 * Represents a training or evaluation set for graph embedding. Iterates over batches and updates model parameters during training.
 */
class DataLoader {
  protected:

    // misc metadata
    bool train_;                                         /**< True if the sampler is a training set */
    int epochs_processed_;                               /**< Total number of epochs that have been trained on this dataset */
    std::atomic<int64_t> batches_processed_;
    int64_t current_edge_;                               /**< ID of the next edge in the dataset which will be processed */
    std::mutex *sampler_lock_;                           /**< Used to prevent race conditions when sampling */
    vector<Batch *> batches_;                            /**< Ordering of the batch objects that will be processed */\

    int batch_id_offset_;
    vector<Batch *>::iterator batch_iterator_;           /**< Iterator for batches_ */
    std::mutex *batch_lock_;                             /**< Mutex for batches_ and batch_iterator_ */
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

    NeighborSampler *training_neighbor_sampler_;
    NeighborSampler *evaluation_neighbor_sampler_;

    NegativeSampler *training_negative_sampler_;
    NegativeSampler *evaluation_negative_sampler_;

    Timestamp timestamp_;                                /**< Timestamp of the current state of the program data */


  public:
    GraphModelStorage *graph_storage_;

    EdgeSampler *edge_sampler_;
    NegativeSampler *negative_sampler_;
    NeighborSampler *neighbor_sampler_;

    shared_ptr<TrainingConfig> training_config_;
    shared_ptr<EvaluationConfig> evaluation_config_;

    /**
     * Constructor
     */
    DataLoader(GraphModelStorage *graph_storage,
               shared_ptr<TrainingConfig> training_config,
               shared_ptr<EvaluationConfig> evaluation_config,
               shared_ptr<EncoderConfig>);

    /**
     * Destructor
     */
    ~DataLoader();

    void setBufferOrdering();

    void setActiveEdges();

    void setActiveNodes();

    void initializeBatches();

    void clearBatches();

    /**
     * Check to see whether another batch exists.
     * @return True if batch exists, false if not
     */
    bool hasNextBatch();

    Batch *getNextBatch();

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
    Batch *getBatch(int worker_id = 0);

    /**
     * Calls getBatch and splits the batch into `num_devices` sub batches. In link prediction the split is done by the edges and in node classification by the nodes.
     */
    std::vector<Batch *> getSubBatches();

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load edges and samples into.
     */
    void linkPredictionSample(Batch *batch, int worker_id = 0);

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load edges and samples into.
     */
    void nodeClassificationSample(Batch *batch, int worker_id = 0);

    /**
     * Loads CPU parameters into batch
     * @param batch: Batch object to load parameters into.
     */
    void loadCPUParameters(Batch *batch);

    /**
     * Loads GPU parameters into batch
     * @param batch Batch object to load parameters into.
     */
    void loadGPUParameters(Batch *batch);

    /**
     * Applies gradient updates to underlying storage
     * @param batch: Batch object to apply updates from.
     * @param gpu: If true, only the gpu parameters will be updated.
     */
    void updateEmbeddingsForBatch(Batch *batch, bool gpu);

    /**
     * Notify that the epoch has been completed. Prepares dataset for a new epoch.
     */
    void nextEpoch();

    /**
     * Load graph from storage.
     */
    void loadStorage();

    bool epochComplete() {
        return (batches_left_ == 0) && all_read_;
    }
    
    /**
     * Unload graph from storage.
     * @param write Set to true to write embedding table state to disk
     */
    void unloadStorage(bool write = false) {
        graph_storage_->unload(write);
    }

    /**
     * Gets the number of edges from the graph storage.
     * @return Number of edges in the graph
     */
    int64_t getNumEdges() {
        return graph_storage_->getNumEdges();
    }

    int64_t getEpochsProcessed() {
        return epochs_processed_;
    }

    int64_t getBatchesProcessed() {
        return batches_processed_;
    }

    bool isTrain() {
        return train_;
    }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to training set.
    */
    void setTrainSet() {
        train_ = true;
        graph_storage_->setTrainSet();
        negative_sampler_ = training_negative_sampler_;
        neighbor_sampler_ = training_neighbor_sampler_;
    }

    /**
     * Sets graph storage, negative sampler, and neighbor sampler to validation set.
    */
    void setValidationSet() {
        train_ = false;
        graph_storage_->setValidationSet();
        negative_sampler_ = evaluation_negative_sampler_;
        neighbor_sampler_ = evaluation_neighbor_sampler_;
    }

    void setTestSet() {
        train_ = false;
        graph_storage_->setTestSet();
        negative_sampler_ = evaluation_negative_sampler_;
        neighbor_sampler_ = evaluation_neighbor_sampler_;
    }
};

#endif //MARIUS_DATASET_H




