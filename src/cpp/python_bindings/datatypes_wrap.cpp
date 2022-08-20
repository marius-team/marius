#include <torch/extension.h>

#include "datatypes.h"

namespace py = pybind11;

void init_datatypes(py::module &m) {

    py::class_<DummyCuda>(m, "DummyCuda")
        .def(py::init<int>(), py::arg("val"))
        .def("start", &DummyCuda::start)
        .def("record", &DummyCuda::record)
        .def("synchronize", &DummyCuda::synchronize)
        .def("elapsed_time", &DummyCuda::elapsed_time);
}