#include "common/pybind_headers.h"

// data
void init_batch(py::module &);
void init_dataloader(py::module &);
void init_graph(py::module &);

// data/samplers
void init_edge_samplers(py::module &);
void init_neg_samplers(py::module &);
void init_neighbor_samplers(py::module &);

PYBIND11_MODULE(_data, m) {
    m.doc() = "Objects for in memory processing and sampling";

    // data/samplers
    auto samplers_m = m.def_submodule("samplers");

    samplers_m.doc() = "Graph Samplers";

    init_edge_samplers(samplers_m);
    init_neg_samplers(samplers_m);
    init_neighbor_samplers(samplers_m);

    // data
    init_batch(m);
    init_dataloader(m);
    init_graph(m);
}
