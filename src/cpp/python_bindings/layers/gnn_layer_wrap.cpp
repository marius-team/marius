//
// Created by Jason Mohoney on 9/30/21.
//
#include <pybind11/pybind11.h>

#include "layers/gnn_layer.h"

namespace py = pybind11;

class PyGNNLayer : GNNLayer {
public:
    using GNNLayer::GNNLayer;
    Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train) override { PYBIND11_OVERRIDE_PURE(Embeddings, GNNLayer, forward, inputs, gnn_graph, train); }
};

void init_gnn_layer(py::module &m) {

    py::class_<GNNLayer, PyGNNLayer>(m, "GNNLayer")
        .def(py::init<>())
        .def_readwrite("input_dim", &GNNLayer::input_dim_)
        .def_readwrite("output_dim", &GNNLayer::output_dim_)
        .def("forward", &GNNLayer::forward, py::arg("inputs"), py::arg("gnn_graph"), py::arg("train"));
}