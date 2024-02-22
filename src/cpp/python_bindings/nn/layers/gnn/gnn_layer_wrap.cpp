//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/layers/gnn/gnn_layer.h"

class PyGNNLayer : GNNLayer {
   public:
    using GNNLayer::GNNLayer;
    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train) override {
        PYBIND11_OVERRIDE_PURE(torch::Tensor, GNNLayer, forward, inputs, dense_graph, train);
    }
};

void init_gnn_layer(py::module &m) {
    py::class_<GNNLayer, PyGNNLayer, Layer, shared_ptr<GNNLayer>>(m, "GNNLayer")
        .def_readwrite("input_dim", &GNNLayer::input_dim_)
        .def_readwrite("output_dim", &GNNLayer::output_dim_)
        .def("forward", &GNNLayer::forward, py::arg("inputs"), py::arg("dense_graph"), py::arg("train"));
}