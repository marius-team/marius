
#ifndef MARIUS_FEATURES_H
#define MARIUS_FEATURES_H

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "common/datatypes.h"
#include "common/util.h"
#include "data/graph.h"

struct FeaturesLoaderConfig {
    string features_type;
    int64_t page_size;
    int64_t feature_dimension;
    int64_t feature_size;
}; 

class FeaturesLoader {
    public:
        shared_ptr<FeaturesLoaderConfig> config_;
        shared_ptr<MariusGraph> graph_;

        virtual ~FeaturesLoader(){};
        virtual int64_t num_pages_for_nodes(torch::Tensor node_ids) = 0;
};

class LinearFeaturesLoader : public FeaturesLoader {
    public:
        int64_t features_per_page_;

        LinearFeaturesLoader(shared_ptr<FeaturesLoaderConfig> config, shared_ptr<MariusGraph> graph);
        int64_t num_pages_for_nodes(torch::Tensor node_ids) override;
};

std::shared_ptr<FeaturesLoader> get_feature_loader(std::shared_ptr<FeaturesLoaderConfig> config, std::shared_ptr<MariusGraph> graph);

#endif  // MARIUS_FEATURES_H