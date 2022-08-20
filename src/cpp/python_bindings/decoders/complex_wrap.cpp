//
// Created by Jason Mohoney on 9/30/21.
//

#include "decoders/complex.h"

#include <torch/extension.h>

void init_complex(py::module &m) {

    // py::arg("tensor_options") = torch::TensorOptions() breaks pymarius import
    torch::python::bind_module<ComplEx>(m, "ComplEx")
        .def(py::init<int, int, torch::TensorOptions, bool>(), py::arg("num_relations"), py::arg("embedding_dim"), py::arg("tensor_options"), py::arg("use_inverse_relations") = true)
        .def("reset", &ComplEx::reset);
}