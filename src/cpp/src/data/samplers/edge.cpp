//
// Created by Jason Mohoney on 2/8/22.
//

#include "data/samplers/edge.h"

RandomEdgeSampler::RandomEdgeSampler(shared_ptr<GraphModelStorage> graph_storage, bool without_replacement) {
    graph_storage_ = graph_storage;
    without_replacement_ = without_replacement;
}

EdgeList RandomEdgeSampler::getEdges(shared_ptr<Batch> batch) {
    return graph_storage_->getEdgesRange(batch->start_idx_, batch->batch_size_).clone().to(torch::kInt64);
}
