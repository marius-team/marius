//
// Created by Jason Mohoney on 2/8/22.
//

#ifndef MARIUS_EDGE_H
#define MARIUS_EDGE_H

#include "storage/graph_storage.h"

/**
 * Samples the edges from a given batch.
 */
class EdgeSampler {
   public:
    shared_ptr<GraphModelStorage> graph_storage_;

    virtual ~EdgeSampler(){};

    /**
     * Get edges for a given batch.
     * @param batch Batch to sample into
     * @return Edges sampled for the batch
     */
    virtual EdgeList getEdges(shared_ptr<Batch> batch) = 0;
};

class RandomEdgeSampler : public EdgeSampler {
   public:
    bool without_replacement_;

    RandomEdgeSampler(shared_ptr<GraphModelStorage> graph_storage, bool without_replacement = true);

    EdgeList getEdges(shared_ptr<Batch> batch) override;
};

#endif  // MARIUS_EDGE_H
