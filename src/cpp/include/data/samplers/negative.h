//
// Created by Jason Mohoney on 2/8/22.
//

#ifndef MARIUS_NEGATIVE_H
#define MARIUS_NEGATIVE_H

#include "storage/graph_storage.h"

std::tuple<torch::Tensor, torch::Tensor> batch_sample(torch::Tensor edges, int num_negatives, bool inverse = false);

torch::Tensor deg_negative_local_filter(torch::Tensor deg_sample_indices, torch::Tensor edges);

torch::Tensor compute_filter_corruption_cpu(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption_gpu(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                            bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                            torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor compute_filter_corruption(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse = false,
                                        bool global = false, LocalFilterMode local_filter_mode = LocalFilterMode::ALL,
                                        torch::Tensor deg_sample_indices = torch::Tensor());

torch::Tensor apply_score_filter(torch::Tensor scores, torch::Tensor filter);

/**
 * Samples the negative edges from a given batch.
 */
class NegativeSampler {
   public:
    virtual ~NegativeSampler(){};

    /**
     * Get negative edges from the given batch.
     * Return a tensor of node ids of shape [num_negs] or a [num_negs, 3] shaped tensor of negative edges.
     * @param inverse Sample for inverse edges
     * @return The negative nodes/edges sampled
     */
    virtual std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges = torch::Tensor(),
                                                                  bool inverse = false) = 0;
};

class CorruptNodeNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    float degree_fraction_;
    bool filtered_;
    LocalFilterMode local_filter_mode_;

    CorruptNodeNegativeSampler(int num_chunks, int num_negatives, float degree_fraction, bool filtered = false,
                               LocalFilterMode local_filter_mode = LocalFilterMode::DEG);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

class CorruptRelNegativeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;
    bool filtered_;

    CorruptRelNegativeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

class NegativeEdgeSampler : public NegativeSampler {
   public:
    int num_chunks_;
    int num_negatives_;

    NegativeEdgeSampler(int num_chunks, int num_negatives, bool filtered = false);

    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges = torch::Tensor(), bool inverse = false) override;
};

#endif  // MARIUS_NEGATIVE_H
