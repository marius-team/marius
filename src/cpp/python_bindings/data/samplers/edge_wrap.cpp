//
// Created by Jason Mohoney on 2/14/22.
//

#include "common/pybind_headers.h"
#include "data/samplers/edge.h"

class PyEdgeSampler : EdgeSampler {
   public:
    using EdgeSampler::EdgeSampler;
    EdgeList getEdges(shared_ptr<Batch> batch) override { PYBIND11_OVERRIDE_PURE_NAME(EdgeList, EdgeSampler, "getEdges", getEdges, batch); }
};

void init_edge_samplers(py::module &m) {
    py::class_<EdgeSampler, PyEdgeSampler, std::shared_ptr<EdgeSampler>>(m, "EdgeSampler")
        .def_readwrite("graph_storage", &EdgeSampler::graph_storage_)
        .def("getEdges", &EdgeSampler::getEdges, py::arg("batch"));

    py::class_<RandomEdgeSampler, EdgeSampler, std::shared_ptr<RandomEdgeSampler>>(m, "RandomEdgeSampler")
        .def_readwrite("without_replacement", &RandomEdgeSampler::without_replacement_)
        .def(py::init<shared_ptr<GraphModelStorage>, bool>(), py::arg("graph_storage"), py::arg("without_replacement") = true);
}