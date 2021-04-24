//
// Created by Jason Mohoney on 7/9/20.
//

#ifndef MARIUS_BATCH_H
#define MARIUS_BATCH_H

#include <datatypes.h>
#include <config.h>
#include <util.h>

using std::vector;


/**
 * Specifies location of the batch in the pipeline
 */
enum class BatchStatus {
    Waiting,
    AccumulatedIndices,
    LoadedEmbeddings,
    TransferredToDevice,
    PreparedForCompute,
    ComputedGradients,
    AccumulatedGradients,
    TransferredToHost,
    Done
};

/**
 * Contains metadata, edges and embeddings for a single batch.
 */
class Batch {
  public:
    int batch_id_;                                          /** ID of the batch */
    int64_t start_idx_;                                     /** Offset in the edges storage */
    int batch_size_;                                        /** Number of edges in the batch */
    bool train_;                                            /** If true, this batch is a training batch and requires gradient tracking */
    int device_id_;                                         /** ID of the device the batch is assigned to */

    Timestamp load_timestamp_;                              /** Timestamp of when the embeddings for the batch have been loaded from storage */
    Timestamp compute_timestamp_;                           /** Timestamp of when the gradients for the batch have been computed */
    CudaEvent device_transfer_;                             /** Used as a sync point when transferring from host to device */
    CudaEvent host_transfer_;                               /** Used as a sync point when transferring from device to host */
    Timer timer_;                                           /** Timer used to track how long batch operations take */
    BatchStatus status_;                                    /** Tracks location of the batch in the pipeline */

    Indices unique_node_indices_;                           /** Global node ids for each unique node in the batch. includes negative samples */
    Embeddings unique_node_embeddings_;                     /** Embedding tensor for each unique node in the the batch.  */
    Gradients unique_node_gradients_;                       /** Gradients for each node embedding in the batch */
    Gradients unique_node_gradients2_;                      /** Node embedding updates after adjusting for optimizer state */
    OptimizerState unique_node_embeddings_state_;           /** Optimizer state for each node embedding in the batch */

    Indices unique_relation_indices_;                       /** Global ids for each unique relation in the batch */
    Relations unique_relation_embeddings_;                  /** Embedding tensor for each relation in the batch. */
    Gradients unique_relation_gradients_;                   /** Gradients for each relation in the batch */
    Gradients unique_relation_gradients2_;                  /** Relation embedding updates after adjusting for optimizer state */
    OptimizerState unique_relation_embeddings_state_;       /** Optimizer state for each relation embedding in the batch */

    Indices src_pos_indices_mapping_;                       /** Maps ids from source nodes in the batch to global node ids */
    Indices dst_pos_indices_mapping_;                       /** Maps ids from destination nodes in the batch to global node ids */
    Indices rel_indices_mapping_;                           /** Maps ids from relations in the batch to global relation ids */
    Indices src_neg_indices_mapping_;                       /** Maps ids from the sampled nodes, which corrupt the source nodes of edges, to global node ids */
    Indices dst_neg_indices_mapping_;                       /** Maps ids from the sampled nodes, which corrupt the destination nodes of edges, to global node ids */

    Indices src_pos_indices_;                               /** Global node ids for source nodes in the batch */
    Indices dst_pos_indices_;                               /** Global node ids for destination nodes in the batch */
    Indices rel_indices_;                                   /** Global relation ids for relations in the batch */
    Indices src_neg_indices_;                               /** Global node ids for the sampled nodes that are used to corrupt the source nodes of edges */
    Indices dst_neg_indices_;                               /** Global node ids for the sampled nodes that are used to corrupt the destination nodes of edges */

    Embeddings src_pos_embeddings_;                         /** Embeddings for the source nodes in the batch */
    Embeddings dst_pos_embeddings_;                         /** Embeddings for the destination nodes in the batch */
    Relations src_relation_emebeddings_;                    /** Embeddings for the relations in the batch */
    Relations dst_relation_emebeddings_;                    /** Embeddings for the relations in the batch */
    Embeddings src_global_neg_embeddings_;                  /** Embeddings for the globally sampled nodes used to corrupt the source nodes of edges in the batch */
    Embeddings dst_global_neg_embeddings_;                  /** Embeddings for the globally sampled nodes used to corrupt the destination nodes of edges in the batch */
    Embeddings src_all_neg_embeddings_;                     /** Embeddings for the globally and locally sampled nodes used to corrupt the source nodes of edges in the batch */
    Embeddings dst_all_neg_embeddings_;                     /** Embeddings for the globally and locally sampled nodes used to corrupt the destination nodes of edges in the batch */

    // Negative Sampling params
    torch::Tensor src_neg_filter_;                          /** Used to filter out false negatives in training for source corrupted negatives */
    torch::Tensor dst_neg_filter_;                          /** Used to filter out false negatives in training for destination corrupted negatives */

    // Evaluation Params
    torch::Tensor ranks_;                                   /** Link prediction ranks for each edge */
    torch::Tensor auc_;                                     /** Link prediction AUC */
    vector<torch::Tensor> src_neg_filter_eval_;             /** Used to filter out false negatives in evaluation for source corrupted negatives */
    vector<torch::Tensor> dst_neg_filter_eval_;             /** Used to filter out false negatives in evaluation for destination corrupted negatives */

    // GNN Neighborhood Params
    SparseAdjacencyMatrixMR adjacency_matrix_;              /** Sparse matrix which holds the adjacency matrix for the batch */

    Batch(bool train);                                      /** Constructor */

    ~Batch() {};                                            /** Destructor */

    void localSample();                                     /** Construct additional negative samples and neighborhood information from the batch */

    virtual void accumulateUniqueIndices();                 /** Populates the unique_<>_indices tensors */

    void embeddingsToDevice(int device_id);                 /** Transfers embeddings, optimizer state, and indices to specified device */

    void prepareBatch();                                    /** Populates the src_pos_embeddings, dst_post_embeddings, relation_embeddings, src_neg_embeddings, and dst_neg_embeddings tensors for model computation */

    virtual void accumulateGradients();                     /** Accumulates gradients into the unique_node_gradients and unique_relation_gradients tensors, and applies optimizer update rule to create the unique_node_gradients2 and unique_relation_gradients2 tensors */

    void embeddingsToHost();                                /** Transfers gradients and embedding updates to host */

    virtual void clear();                                   /** Clears all tensor data in the batch */

};

/**
 * Used when training with partitions.
 */
class PartitionBatch : public Batch {
  public:
    int64_t src_partition_idx_;                             /** Id of the source partition */
    int64_t dst_partition_idx_;                             /** Id of the destination partition */

    Indices pos_uniques_idx_;                               /** Partition node ids for each unique node in the batch. Does not include negative samples */
    Indices src_pos_uniques_idx_;                           /** Partition node ids for each unique source node in the batch. Does not include negative samples */
    Indices dst_pos_uniques_idx_;                           /** Partition node ids for each unique destination node in the batch. Does not include negative samples */
    Indices neg_uniques_idx_;                               /** Global node ids for sampled nodes which produce negatives */

    std::vector<int> buffer_state_;                         /** State of the buffer when this batch was read, used to check for evicited negatives */

    PartitionBatch(bool train);                             /** Constructor */

    ~PartitionBatch() {};                                   /** Destructor */

    void accumulateUniqueIndices() override;                /** Populates the uniques tensors */

    void clear() override;                                  /** Clears all tensor data in the batch */

};

#endif //MARIUS_BATCH_H
