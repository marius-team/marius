#include "encoders/gnn.h"

#include <torch/extension.h>

void init_gnn(py::module &m) {

    torch::python::bind_module<GeneralGNN>(m, "GeneralGNN")
        .def_readwrite("encoder_config", &GeneralGNN::encoder_config_)
        .def_readwrite("num_relations", &GeneralGNN::num_relations_)
        .def_readwrite("device", &GeneralGNN::device_)
        .def_readwrite("layers", &GeneralGNN::layers_)
        .def(py::init<shared_ptr<EncoderConfig>, torch::Device, int>(),
            py::arg("encoder_config"),
            py::arg("device"),
            py::arg("num_relations") = 1)
        .def("forward", &GeneralGNN::forward,
            py::arg("inputs"),
            py::arg("gnn_graph"),
            py::arg("train") = true)
        .def("reset", &GeneralGNN::reset);
}