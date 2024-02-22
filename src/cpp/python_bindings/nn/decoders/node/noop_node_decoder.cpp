//
// Created by Jason Mohoney on 2/15/22.
//

#include "nn/decoders/node/noop_node_decoder.h"

#include <common/pybind_headers.h>

void init_noop_node_decoder(py::module &m) {
    py::class_<NoOpNodeDecoder, NodeDecoder, torch::nn::Module, shared_ptr<NoOpNodeDecoder>>(m, "NoOpNodeDecoder")
        .def(py::init<>())
        .def("compute_labels", &NoOpNodeDecoder::forward, py::arg("nodes"))
        .def("reset", &NoOpNodeDecoder::reset);
}
