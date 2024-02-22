//
// Created by Jason Mohoney on 7/9/20.
//

#ifndef MARIUS_BATCH_H
#define MARIUS_BATCH_H

#include "common/datatypes.h"
#include "common/util.h"
#include "graph.h"

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
    int batch_id_;      /**< ID of the batch */
    int64_t start_idx_; /**< Offset in the edges storage */
    int batch_size_;    /**< Number of edges in the batch */
    bool train_;        /**< If true, this batch is a training batch and requires gradient tracking */
    int device_id_;     /**< ID of the device the batch is assigned to */

    LearningTask task_;

    Timestamp load_timestamp_;    /**< Timestamp of when the embeddings for the batch have been loaded from storage */
    Timestamp compute_timestamp_; /**< Timestamp of when the gradients for the batch have been computed */
    CudaEvent device_transfer_;   /**< Used as a sync point when transferring from host to device */
    CudaEvent host_transfer_;     /**< Used as a sync point when transferring from device to host */
    Timer timer_;                 /**< Timer used to track how long batch operations take */
    BatchStatus status_;          /**< Tracks location of the batch in the pipeline */

    Indices root_node_indices_;
    Indices unique_node_indices_;         /**< Global node ids for each unique node in the batch. includes negative samples */
    torch::Tensor node_embeddings_;       /**< Embedding tensor for each unique node in the the batch.  */
    torch::Tensor node_gradients_;        /**< Gradients for each node embedding in the batch */
    torch::Tensor node_embeddings_state_; /**< Optimizer state for each node embedding in the batch */
    torch::Tensor node_state_update_;     /**< Updates to adjust the optimizer state */

    torch::Tensor node_features_; /**< Feature vector for each unique node in the the batch.  */
    torch::Tensor node_labels_;   /**< Label for each unique node in the the batch.  */

    Indices src_neg_indices_mapping_; /**< Maps ids from the sampled nodes, which corrupt the source nodes of edges, to global node ids */
    Indices dst_neg_indices_mapping_; /**< Maps ids from the sampled nodes, which corrupt the destination nodes of edges, to global node ids */

    torch::Tensor edges_;

    // Encoder
    DENSEGraph dense_graph_;
    torch::Tensor encoded_uniques_;

    // Negative Sampling params
    torch::Tensor neg_edges_;
    Indices rel_neg_indices_; /**< Global relation ids for negative relations in the batch */
    Indices src_neg_indices_; /**< Global node ids for the sampled nodes that are used to corrupt the source nodes of edges */
    Indices dst_neg_indices_; /**< Global node ids for the sampled nodes that are used to corrupt the destination nodes of edges */

    torch::Tensor src_neg_filter_; /**< Used to filter out false negatives for source corrupted negatives */
    torch::Tensor dst_neg_filter_; /**< Used to filter out false negatives for destination corrupted negatives */

    Batch(bool train); /**< Constructor */

    ~Batch(); /**< Destructor */

    void to(torch::Device device, CudaStream *compute_stream = nullptr); /**< Transfers embeddings, optimizer state, and indices to specified device */

    void accumulateGradients(float learning_rate); /**< Accumulates gradients into the unique_node_gradients, and applies optimizer update rule to create the
                                                      unique_node_gradients2 tensor */

    void embeddingsToHost(); /**< Transfers gradients and embedding updates to host */

    void clear(); /**< Clears all tensor data in the batch */
};
#endif  // MARIUS_BATCH_H
