//
// Created by Jason Mohoney on 9/30/21.
//

#include "layers/gat_layer.h"

#include <torch/extension.h>

void init_gat_layer(py::module &m) {

     torch::python::bind_module<GATLayer>(m, "GATLayer")
          .def_readwrite("layer_config", &GATLayer::layer_config_)
          .def_readwrite("options", &GATLayer::options_)
          .def_readwrite("head_dim", &GATLayer::head_dim_)
          .def_readwrite("input_dropout", &GATLayer::input_dropout_)
          .def_readwrite("attention_dropout", &GATLayer::attention_dropout_)
          .def_readwrite("weight_matrices", &GATLayer::weight_matrices_)
          .def_readwrite("a_l", &GATLayer::a_l_)
          .def_readwrite("a_r", &GATLayer::a_r_)
          .def_readwrite("bias", &GATLayer::bias_)
          .def_readwrite("device", &GATLayer::device_)
          .def(py::init<shared_ptr<GNNLayerConfig>, torch::DeviceType>(),
               py::arg("layer_config"),
               py::arg("device"))
          .def("reset", &GATLayer::reset)
          .def("forward", &GATLayer::forward,
               py::arg("inputs"),
               py::arg("gnn_graph"),
               py::arg("train") = true);
}