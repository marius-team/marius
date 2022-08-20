//
// Created by Jason Mohoney on 9/30/21.
//

#include "layers/gcn_layer.h"

#include <torch/extension.h>

void init_gcn_layer(py::module &m) {

     torch::python::bind_module<GCNLayer>(m, "GCNLayer")
          .def_readwrite("layer_config", &GCNLayer::layer_config_)
          .def_readwrite("options", &GCNLayer::options_)
          .def_readwrite("use_incoming", &GCNLayer::use_incoming_)
          .def_readwrite("use_outgoing", &GCNLayer::use_outgoing_)
          .def_readwrite("w", &GCNLayer::w_)
          .def_readwrite("bias", &GCNLayer::bias_)
          .def_readwrite("device", &GCNLayer::device_)
          .def(py::init<shared_ptr<GNNLayerConfig>, bool, bool, torch::DeviceType>(),
               py::arg("layer_config"),
               py::arg("use_incoming"),
               py::arg("use_outgoing"),
               py::arg("device"))
          .def("reset", &GCNLayer::reset)
          .def("forward", &GCNLayer::forward,
               py::arg("inputs"),
               py::arg("gnn_graph"),
               py::arg("train") = true);
}