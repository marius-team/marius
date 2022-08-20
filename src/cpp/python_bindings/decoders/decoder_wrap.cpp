//
// Created by Jason Mohoney on 9/30/21.
//

#include "decoders/decoder.h"

#include <torch/extension.h>

class PyDecoder : Decoder {
public:
    using Decoder::Decoder;
    using ReturnTensorTuple = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
    ReturnTensorTuple forward(Batch *batch, bool train) override {
        PYBIND11_OVERRIDE_PURE(ReturnTensorTuple, Decoder, forward, batch, train); }
};

void init_decoder(py::module &m) {

    py::class_<Decoder, PyDecoder>(m, "Decoder")
        .def(py::init<>())
        .def("forward", &Decoder::forward, py::arg("batch"), py::arg("train"));

    torch::python::bind_module<EmptyDecoder>(m, "EmptyDecoder")
        .def(py::init<>())
        .def("reset", &EmptyDecoder::reset);
    
    py::class_<LinkPredictionDecoder, Decoder>(m, "LinkPredictionDecoder")
        .def_readwrite("use_inverse_relations_", &LinkPredictionDecoder::use_inverse_relations_)
        .def(py::init<>());
}