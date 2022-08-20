//
// Created by Jason Mohoney on 9/30/21.
//
#include "decoders/transe.h"

#include <torch/extension.h>

void init_transe(py::module &m) {

    torch::python::bind_module<TransE>(m, "TransE")
            .def(py::init<int, int, torch::TensorOptions, bool>(), py::arg("num_relations"), py::arg("embedding_dim"), py::arg("tensor_options"), py::arg("use_inverse_relations") = true)
            .def("reset", &TransE::reset);
}