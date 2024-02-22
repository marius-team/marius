//
// Created by Jason Mohoney on 4/9/21.
//

#include "common/pybind_headers.h"
#include "marius.h"

void init_marius(py::module &m) {
    m.def("marius_train", &marius_train, py::arg("config"), py::call_guard<py::gil_scoped_release>());
    m.def("marius_eval", &marius_eval, py::arg("config"), py::call_guard<py::gil_scoped_release>());
}
