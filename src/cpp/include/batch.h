//
// Created by Jason Mohoney on 7/9/20.
//

#ifndef MARIUS_BATCH_H
#define MARIUS_BATCH_H

#include "datatypes.h"
#include "graph.h"
#include "util.h"

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
    int batch_id_;                                          /**< ID of the batch */
    int64_t start_idx_;                                     /**< Offset in the edges storage */
    int batch_size_;                                        /**< Number of edges in the batch */
    bool train_;                                            /**< If true, this batch is a training batch and requires gradient tracking */
    int device_id_;                                         /**< ID of the device the batch is assigned to */

    Timestamp load_timestamp_;                              /**< Timestamp of when the embeddings for the batch have been loaded from storage */
    Timestamp compute_timestamp_;                           /**< Timestamp of when the gradients for the batch have been computed */
    CudaEvent device_transfer_;                             /**< Used as a sync point when transferring from host to device */
    CudaEvent host_transfer_;                               /**< Used as a sync point when transferring from device to host */
    Timer timer_;                                           /**< Timer used to track how long batch operations take */
    BatchStatus status_;                                    /**< Tracks location of the batch in the pipeline */

    Indices root_node_indices_;
    Indices unique_node_indices_;                           /**< Global node ids for each unique node in the batch. includes negative samples */
    Embeddings unique_node_embeddings_;                     /**< Embedding tensor for each unique node in the the batch.  */
    Gradients unique_node_gradients_;                       /**< Gradients for each node embedding in the batch */
    OptimizerState unique_node_embeddings_state_;           /**< Optimizer state for each node embedding in the batch */
    Gradients unique_node_state_update_;                    /**< Updates to adjust the optimizer state */

    Features unique_node_features_;                         /**< Embedding vector for each unique node in the the batch.  */
    Labels unique_node_labels_;                             /**< Label for each unique node in the the batch.  */

    Indices src_pos_indices_mapping_;                       /**< Maps ids from source nodes in the batch to global node ids */
    Indices dst_pos_indices_mapping_;                       /**< Maps ids from destination nodes in the batch to global node ids */
    Indices src_neg_indices_mapping_;                       /**< Maps ids from the sampled nodes, which corrupt the source nodes of edges, to global node ids */
    Indices dst_neg_indices_mapping_;                       /**< Maps ids from the sampled nodes, which corrupt the destination nodes of edges, to global node ids */

    Indices src_pos_indices_;                               /**< Global node ids for source nodes in the batch */
    Indices dst_pos_indices_;                               /**< Global node ids for destination nodes in the batch */
    Indices rel_indices_;                                   /**< Global relation ids for relations in the batch */
    Indices src_neg_indices_;                               /**< Global node ids for the sampled nodes that are used to corrupt the source nodes of edges */
    Indices dst_neg_indices_;                               /**< Global node ids for the sampled nodes that are used to corrupt the destination nodes of edges */

    shared_ptr<NegativeSamplingConfig> negative_sampling_;

    // Encoder
    GNNGraph gnn_graph_;
    Embeddings encoded_uniques_;

    // Decoder
    Embeddings src_pos_embeddings_;                         /**< Embeddings for the source nodes in the batch */
    Embeddings dst_pos_embeddings_;                         /**< Embeddings for the destination nodes in the batch */
    Embeddings src_global_neg_embeddings_;                  /**< Embeddings for the globally sampled nodes used to corrupt the source nodes of edges in the batch */
    Embeddings dst_global_neg_embeddings_;                  /**< Embeddings for the globally sampled nodes used to corrupt the destination nodes of edges in the batch */
    Embeddings src_all_neg_embeddings_;                     /**< Embeddings for the globally and locally sampled nodes used to corrupt the source nodes of edges in the batch */
    Embeddings dst_all_neg_embeddings_;                     /**< Embeddings for the globally and locally sampled nodes used to corrupt the destination nodes of edges in the batch */

    // Negative Sampling params
    torch::Tensor src_neg_filter_;                          /**< Used to filter out false negatives in training for source corrupted negatives */
    torch::Tensor dst_neg_filter_;                          /**< Used to filter out false negatives in training for destination corrupted negatives */

    vector<torch::Tensor> src_neg_filter_eval_;             /**< Used to filter out false negatives in evaluation for source corrupted negatives */
    vector<torch::Tensor> dst_neg_filter_eval_;             /**< Used to filter out false negatives in evaluation for destination corrupted negatives */

    Batch(bool train);                                      /**< Constructor */

    Batch(std::vector<Batch *> sub_batches);                /**< Merges multiple sub batches with node embedding gradients and optimizer state into a single batch of gradients. Use for accumulating gradients in a multi-GPU setting.*/

    ~Batch();                                               /**< Destructor */

    void setUniqueNodes(bool use_neighbors = false, bool set_mapping = false);

    void localSample();                                     /**< Sample additional negatives from the batch */

    void to(torch::Device device, at::cuda::CUDAStream *compute_stream = nullptr);                          /**< Transfers embeddings, optimizer state, and indices to specified device */

    void prepareBatch();                                    /**< Populates the src_pos_embeddings, dst_post_embeddings, src_neg_embeddings, and dst_neg_embeddings tensors for model computation */

    void accumulateGradients(float learning_rate);          /**< Accumulates gradients into the unique_node_gradients, and applies optimizer update rule to create the unique_node_gradients2 tensor */

    void embeddingsToHost();                                /**< Transfers gradients and embedding updates to host */

    void clear();                                           /**< Clears all tensor data in the batch */

};
#endif //MARIUS_BATCH_H
