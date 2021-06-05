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
#include "config.h"
#include "datatypes.h"
#include "logger.h"
#include "storage.h"

using std::map;
using std::vector;
using std::tuple;
using std::mutex;
using std::pair;
using std::forward_as_tuple;
using std::unique_ptr;

/**
 * Represents a training or evaluation set for graph embedding. Iterates over batches and updates model parameters during training.
 */
class DataSet {
  protected:
    // misc metadata
    bool train_;                                         /** True if the dataset is a training set */
    int epochs_processed_;                               /** Total number of epochs that have been trained on this dataset */
    int64_t num_edges_;                                  /** Total number of edges in this dataset */
    int64_t num_nodes_;                                  /** Total number of nodes in this dataset */
    int64_t num_relations_;                              /** Total number of relations (edge-types) in this dataset */
    int64_t current_edge_;                               /** ID of the next edge in the dataset which will be processed */
    int64_t current_negative_id_;                        /** Next negative node to be sampled (only used for shuffled negative sampling) */
    mutex *negative_lock_;                               /** Used to prevent race conditions when sampling negatives */

    // batch ordering
    vector<int64_t> edge_bucket_sizes_;                  /** Total number of edges in each edge bucket */
    vector<Batch *> batches_;                            /** Ordering of the batch objects that will be processed */
    vector<Batch *>::iterator batch_iterator_;           /** Iterator for batches_ */
    mutex *batch_lock_;                                  /** Mutex for batches_ and batch_iterator_ */

    // used for evaluation
    SparseAdjacencyMatrixMR adjacency_matrix_;           /** TODO Sparse adjancency matrix used to filter false negatives in evaluation */
    map<pair<int, int>, vector<int>> src_map_;           /** Map keyed by the source node and relation ids, where the destination node is the value. Provides fast lookups for edge existence */
    map<pair<int, int>, vector<int>> dst_map_;           /** Map keyed by the destination node and relation ids, where the source node is the value. Provides fast lookups for edge existence */

    // Program data storage
    bool storage_loaded_;
    Storage *edges_;                                     /** Pointer to storage of the edges that are currently being processed */
    Storage *train_edges_;                               /** Pointer to storage of the training set edges */
    Storage *validation_edges_;                          /** Pointer to storage of the validation set edges */
    Storage *test_edges_;                                /** Pointer to storage of the test set edges */
    Storage *node_embeddings_;                           /** Pointer to storage of the node embeddings */
    Storage *node_embeddings_optimizer_state_;           /** Pointer to storage of the node embedding optimizer state */
    Storage *src_relations_;                             /** Pointer to storage of the relation embeddings applied to the source nodes of an edge */
    Storage *src_relations_optimizer_state_;             /** Pointer to storage of the optimizer state for relation embeddings applied to the source nodes of an edge */
    Storage *dst_relations_;                             /** Pointer to storage of the relation embeddings applied to the destination nodes of an edge */
    Storage *dst_relations_optimizer_state_;             /** Pointer to storage of the optimizer state for relation embeddings applied to the source nodes of an edge */

    Timestamp timestamp_;                                /** Timestamp of the current state of the program data */

  public:
    /**
     * Training set constructor
     */
    DataSet(Storage *edges, Storage *embeddings, Storage *emb_state, Storage *src_relations, Storage *src_rel_state, Storage *dst_relations, Storage *dst_rel_state);

    /**
     * Evaluation set constructor
     */
    DataSet(Storage *train_edges, Storage *eval_edges, Storage *test_edges, Storage *embeddings, Storage *src_relations, Storage *dst_relations);

    /**
     * Evaluation set constructor for separate evaluation without training
     */
    DataSet(Storage *test_edges, Storage *embeddings, Storage *src_relations, Storage *dst_relations);

    /**
     * Destructor
     */
     ~DataSet();

    /**
     * Returns the next uninitialized batch from the batch_iterator_ in a thread safe manner
     * @return Pointer to the next batch object to be processed
     */
    Batch *nextBatch();

    /**
     * Initializes the batches for a single epoch
     */
    void initializeBatches();

    /**
     * Splits edge buckets into batches
     */
    void splitBatches();

    /**
     * Clears the resources held by the batches_ list
     */
    void clearBatches();

    /**
     * Sets the src_neg_filter_eval_ and dst_neg_filter_eval_ class members of the batch object, which will be used to filter false negatives in evaluation
     * @param batch: Batch object to set filters for.
     */
    void setEvalFilter(Batch *batch);

    /**
     * Samples num_chunks * negatives uniformly from the range [0, num_nodes)
     * @return A [num_chunks, negatives] sized indices tensor
     */
    Indices uniformIndices();

    /**
     * Samples num_chunks * negatives uniformly from the range [0, partition_size)
     * @return A [num_chunks, negatives] sized indices tensor
     */
    Indices uniformIndices(int partition_id);

    /**
     * Samples num_chunks * negatives sequentially from current_negative_id_
     * @return A [num_chunks, negatives] sized indices tensor
     */
    Indices shuffleIndices();

    /**
     * Calls the negative sampler specified by the config. Samples globally
     * @return A [num_chunks, negatives] sized indices tensor
     */
    Indices getNegativesIndices();

    /**
     * Calls the negative sampler specified by the config. Samples from within the specified partition
     * @return A [num_chunks, negatives] sized indices tensor
     */
    Indices getNegativesIndices(int partition_id);

    /**
     * Gets the next batch to be processed by the pipeline.
     * Loads edges from storage
     * Constructs negative negative edges
     * Loads CPU embedding parameters
     */
    Batch *getBatch();

    /**
     * Loads edges and samples negatives to construct a batch
     * @param batch: Batch object to load edges and samples into.
     */
    void globalSample(Batch *batch);

    /**
     * Loads CPU parameters into batch
     * @param batch: Batch object to load parameters into.
     */
    void loadCPUParameters(Batch *batch);

    /**
     * Loads GPU parameters into batch
     * @param batch: Batch object to load parameters into.
     */
    void loadGPUParameters(Batch *batch);

    /**
     * Applies gradient updates to underlying storage
     * @param batch: Batch object to apply updates from.
     * @param gpu: If true, only the gpu parameters will be updated.
     */
    void updateEmbeddingsForBatch(Batch *batch, bool gpu);

    /**
     * Prepares dataset for a new epoch
     */
    void nextEpoch();

    /**
     * Create checkpoint files for the current state of the parameters in the experiment directory
     */
    void checkpointParameters();

    /**
     * Loads necessary program data from storage to prepare for dataset usage.
     */
     void loadStorage();

    /**
    * Unloads storage data from memory
    */
    void unloadStorage();

    /**
     * @return Returns true if there are batches left to process, if false there are no batches left to process and the epoch is complete
     */
    bool hasNextBatch() {
        batch_lock_->lock();
        bool ret = batch_iterator_ != batches_.end();
        batch_lock_->unlock();
        return ret;
    }

    bool isDone() {
        return batches_processed_ == (int64_t) batches_.size();
    }

    void setTestSet() {
        edges_->unload();
        edges_ = test_edges_;
        edges_->load();
        num_edges_ = edges_->getDim0();
        clearBatches();
        initializeBatches();
        batch_iterator_ = batches_.begin();
    }

    void setValidationSet() {
        edges_->unload();
        edges_ = validation_edges_;
        edges_->load();
        num_edges_ = edges_->getDim0();
        clearBatches();
        initializeBatches();
        batch_iterator_ = batches_.begin();
    }

    torch::DeviceType getDevice() {
        return marius_options.general.device;
    }

    int64_t getEpochsProcessed() {
        return epochs_processed_;
    }

    int64_t getNumEdges() {
        return num_edges_;
    }

    int64_t getNumBatches() {
        return batches_.size();
    }

    int64_t getNumNodes() {
        return num_nodes_;
    }

    int64_t getBatchesProcessed() {
        return batches_processed_;
    }

    tuple<int64_t, int64_t, float> getProgress() {
        return forward_as_tuple(current_edge_, num_edges_, (float) current_edge_ / num_edges_);
    }

    Timestamp getTimestamp() {
        return timestamp_;
    }

    void updateTimestamp() {
        timestamp_ = global_timestamp_allocator.getTimestamp();
    }

    bool isTrain() {
        return train_;
    }

    void setCurrPos(int64_t curr_pos) {
        current_edge_ = curr_pos;
    }

    void syncEmbeddings();

    torch::Tensor accumulateRanks();

    float accumulateAuc();

    std::atomic<int64_t> batches_processed_;
};

#endif //MARIUS_DATASET_H




