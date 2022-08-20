#include <torch/extension.h>

#include "graph_samplers.h"

namespace py = pybind11;

class PyEdgeSampler : EdgeSampler {
  public:
    using EdgeSampler::EdgeSampler;
    EdgeList getEdges(Batch *batch) override {
        PYBIND11_OVERRIDE_PURE_NAME(EdgeList, EdgeSampler, "getEdges", getEdges, batch); }
};

class PyNegativeSampler : NegativeSampler {
  public:
    using NegativeSampler::NegativeSampler;
    EdgeList getNegatives(Batch *batch, bool src) override {
        PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, NegativeSampler, "getNegatives", getNegatives, batch, src); }
};

class PyNeighborSampler : NeighborSampler {
  public:
    using NeighborSampler::NeighborSampler;
    GNNGraph getNeighbors(torch::Tensor node_ids) override {
        PYBIND11_OVERRIDE_PURE_NAME(GNNGraph, NeighborSampler, "getNeighbors", getNeighbors, node_ids); }
};

void init_graph_samplers(py::module &m) {

    py::class_<EdgeSampler, PyEdgeSampler>(m, "EdgeSampler")
        .def_readwrite("graph_storage", &EdgeSampler::graph_storage_)
        .def("getEdges", &EdgeSampler::getEdges, py::arg("batch"));

    py::class_<RandomEdgeSampler, EdgeSampler>(m, "RandomEdgeSampler")
        .def_readwrite("without_replacement", &RandomEdgeSampler::without_replacement_)
        .def(py::init<GraphModelStorage *, bool>(), py::arg("graph_storage"), py::arg("without_replacement") = true);

    py::class_<NegativeSampler, PyNegativeSampler>(m, "NegativeSampler")
        .def_readwrite("graph_storage", &NegativeSampler::graph_storage_)
        //.def_readwrite("sampler_lock", &NegativeSampler::sampler_lock_)
        .def("getNegatives", &NegativeSampler::getNegatives, py::arg("batch"), py::arg("src") = true)
        .def("lock", &NegativeSampler::lock)
        .def("unlock", &NegativeSampler::unlock);

    py::class_<RandomNegativeSampler, NegativeSampler>(m, "RandomNegativeSampler")
        .def_readwrite("without_replacement", &RandomNegativeSampler::without_replacement_)
        .def_readwrite("num_chunks", &RandomNegativeSampler::num_chunks_)
        .def_readwrite("num_negatives", &RandomNegativeSampler::num_negatives_)
        .def(py::init<GraphModelStorage *, int, int, bool>(), 
            py::arg("graph_storage"), 
            py::arg("num_chunks"),
            py::arg("num_negatives"),
            py::arg("without_replacement") = true);
    
    py::class_<FilteredNegativeSampler, NegativeSampler>(m, "FilteredNegativeSampler")
        .def(py::init<GraphModelStorage *>(), py::arg("graph_storage"));

    py::class_<NeighborSampler, PyNeighborSampler>(m, "NeighborSampler")
        .def_readwrite("storage", &NeighborSampler::storage_)
        .def_readwrite("incoming", &NeighborSampler::incoming_)
        .def_readwrite("outgoing", &NeighborSampler::outgoing_)
        .def("getNeighbors", &NeighborSampler::getNeighbors, py::arg("node_ids"));

//    py::class_<LayeredNeighborSampler, NeighborSampler>(m, "LayeredNeighborSampler")
//        .def_readwrite("sampling_layers", &LayeredNeighborSampler::sampling_layers_)
//        .def(py::init<GraphModelStorage *, std::vector<shared_ptr<NeighborSamplingConfig>>, bool, bool>(),
//            py::arg("storage"),
//            py::arg("layer_configs"),
//            py::arg("incoming"),
//            py::arg("outgoing"));

    py::class_<LayeredNeighborSampler, NeighborSampler>(m, "LayeredNeighborSampler")
            .def_readwrite("sampling_layers", &LayeredNeighborSampler::sampling_layers_)
            .def(py::init([](GraphModelStorage *storage, std::vector<int> num_neighbors, bool incoming, bool outgoing, bool use_hashmap_sets) {

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
                    sampling_layers.emplace_back(ptr);
                }
                return std::unique_ptr<LayeredNeighborSampler>(new LayeredNeighborSampler(storage, sampling_layers, incoming, outgoing, use_hashmap_sets));
            }));
}