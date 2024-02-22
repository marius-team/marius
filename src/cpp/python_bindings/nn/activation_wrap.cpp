#include "common/pybind_headers.h"
#include "nn/activation.h"

namespace py = pybind11;

void init_activation(py::module &m) { m.def("apply_activation", &apply_activation, py::arg("activation_function"), py::arg("input")); }