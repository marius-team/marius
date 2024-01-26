//
// Created by Jason Mohoney on 2/8/22.
//

#include "data/samplers/edge.h"

RandomEdgeSampler::RandomEdgeSampler(shared_ptr<GraphModelStorage> graph_storage, bool without_replacement) {
    graph_storage_ = graph_storage;
    without_replacement_ = without_replacement;
}

std::vector<EdgeList> RandomEdgeSampler::getEdges(shared_ptr<Batch> batch) {
    std::vector<EdgeList> sampled_edges;
    sampled_edges.push_back(graph_storage_->getEdgesRange(batch->start_idx_, batch->batch_size_).clone().to(torch::kInt64));
    if(graph_storage_->hasEdgeWeights()) {
        sampled_edges.push_back(graph_storage_->getEdgesWeightsRange(batch->start_idx_, batch->batch_size_).clone().to(torch::kFloat32));
    }
    return sampled_edges;
}
