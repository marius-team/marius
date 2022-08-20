//
// Created by Jason Mohoney on 9/30/21.
//

// #include "encoders/encoder.h"

// #include <torch/extension.h>

// class PyEncoder : Encoder {
// public:
//     using Encoder::Encoder;
//     Embeddings forward(Embeddings inputs, GNNGraph gnn_graph, bool train) override {
//         PYBIND11_OVERRIDE_PURE(Embeddings, Encoder, forward, inputs, gnn_graph, train); }
// };

// void init_encoder(py::module &m) {

//     py::class_<Encoder, PyEncoder>(m, "Encoder")
//             .def(py::init<>())
//             .def("forward", &Encoder::forward, py::arg("inputs"), py::arg("gnn_graph"), py::arg("train"));

//         .def("forward", &Encoder::forward, py::arg("inputs"), py::arg("gnn_graph"), py::arg("train"));

//     torch::python::bind_module<EmptyEncoder>(m, "EmptyEncoder")
//         .def(py::init<>())
//         .def("reset", &EmptyEncoder::reset);

// }