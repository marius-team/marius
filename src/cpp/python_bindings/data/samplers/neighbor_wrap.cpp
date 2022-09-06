//
// Created by Jason Mohoney on 2/14/22.
//

#include "common/pybind_headers.h"
#include "data/samplers/neighbor.h"

class PyNeighborSampler : NeighborSampler {
   public:
    using NeighborSampler::NeighborSampler;
    DENSEGraph getNeighbors(torch::Tensor node_ids, shared_ptr<MariusGraph> graph) override {
        PYBIND11_OVERRIDE_PURE_NAME(DENSEGraph, NeighborSampler, "getNeighbors", getNeighbors, node_ids, graph);
    }
};

void init_neighbor_samplers(py::module &m) {
    py::class_<NeighborSampler, PyNeighborSampler, std::shared_ptr<NeighborSampler>>(m, "NeighborSampler")
        .def_readwrite("storage", &NeighborSampler::storage_)
        .def("getNeighbors", &NeighborSampler::getNeighbors, py::arg("node_ids"), py::arg("graph") = nullptr);

    py::class_<LayeredNeighborSampler, NeighborSampler, std::shared_ptr<LayeredNeighborSampler>>(m, "LayeredNeighborSampler")
        .def_readwrite("sampling_layers", &LayeredNeighborSampler::sampling_layers_)

        .def(py::init([](shared_ptr<GraphModelStorage> storage, std::vector<int> num_neighbors, bool incoming, bool outgoing, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;
                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_incoming_nbrs = incoming;
                     ptr->use_outgoing_nbrs = outgoing;
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(storage, sampling_layers);
             }),
             py::arg("storage"), py::arg("num_neighbors"), py::arg("incoming") = true, py::arg("outgoing") = true, py::arg("use_hashmap_sets") = false)

        .def(py::init([](shared_ptr<MariusGraph> graph, std::vector<int> num_neighbors, bool incoming, bool outgoing, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;

                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_incoming_nbrs = incoming;
                     ptr->use_outgoing_nbrs = outgoing;
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(graph, sampling_layers);
             }),
             py::arg("graph"), py::arg("num_neighbors"), py::arg("incoming") = true, py::arg("outgoing") = true, py::arg("use_hashmap_sets") = false)

        .def(py::init([](std::vector<int> num_neighbors, bool incoming, bool outgoing, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;

                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_incoming_nbrs = incoming;
                     ptr->use_outgoing_nbrs = outgoing;
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(sampling_layers);
             }),
             py::arg("num_neighbors"), py::arg("incoming") = true, py::arg("outgoing") = true, py::arg("use_hashmap_sets") = false);
}
