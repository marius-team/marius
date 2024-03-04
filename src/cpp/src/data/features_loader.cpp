
#include "data/features_loader.h"
#include "common/util.h"

#include <cmath>

LinearFeaturesLoader::LinearFeaturesLoader(shared_ptr<FeaturesLoaderConfig> config, shared_ptr<MariusGraph> graph) {
    config_ = config;
    graph_ = graph;
    features_per_page_ = floor(config->page_size/(config->feature_dimension * config->feature_size));
    std::cout << "Have a total of " << features_per_page_ << " node features per page" << std::endl;
}

int64_t LinearFeaturesLoader::num_pages_for_nodes(torch::Tensor node_ids) {
    torch::Tensor node_pages = torch::floor_divide(node_ids, features_per_page_);
    torch::Tensor unique_pages = std::get<0>(at::_unique(node_pages));
    return unique_pages.numel();
}

shared_ptr<FeaturesLoader> get_feature_loader(shared_ptr<FeaturesLoaderConfig> config, shared_ptr<MariusGraph> graph) {
    if(config->features_type == "linear") {
        return std::make_shared<LinearFeaturesLoader>(config, graph);
    }
    throw std::runtime_error("Invalid feature loader type of " + config->features_type);
}