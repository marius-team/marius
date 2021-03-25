//
// Created by Jason Mohoney on 3/23/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <model.h>

namespace py = pybind11;

void init_model(py::module &m) {
    py::class_<Model>(m, "Model")
        .def(py::init<Encoder *, Decoder *>(), py::arg("encoder"), py::arg("decoder"))
        .def("train", &Model::train, py::arg("batch"))
        .def("evaluate", &Model::evaluate, py::arg("batch"));

    m.def("initializeModel", &initializeModel);
}