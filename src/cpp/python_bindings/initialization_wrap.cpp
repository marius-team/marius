#include <torch/extension.h>

#include "initialization.h"

namespace py = pybind11;

void init_initialization(py::module &m) {

    m.def("compute_fans", &compute_fans, py::arg("shape"));

    m.def("glorot_uniform", &glorot_uniform, py::arg("shape"), py::arg("fans"), py::arg("options"));

    m.def("glorot_normal", &glorot_normal, py::arg("shape"), py::arg("fans"), py::arg("options"));

    m.def("constant_init", &constant_init, py::arg("constant"), py::arg("shape"), py::arg("options"));

    m.def("uniform_init", &uniform_init, py::arg("scale_factor"), py::arg("shape"), py::arg("options"));

    m.def("normal_init", &normal_init, py::arg("mean"), py::arg("std"), py::arg("shape"), py::arg("options"));

    m.def("initialize_tensor", &initialize_tensor, py::arg("init_config"), py::arg("shape"), py::arg("tensor_options"), py::arg("fans"));

    m.def("initialize_subtensor", &initialize_subtensor, py::arg("init_config"), py::arg("sub_shape"), py::arg("full_shape"), py::arg("tensor_options"), py::arg("fans"));
}