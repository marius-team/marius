//
// Created by Jason Mohoney on 9/30/21.
//

#include "layers/rgcn_layer.h"

#include <torch/extension.h>

void init_rgcn_layer(py::module &m) {

     torch::python::bind_module<RGCNLayer>(m, "RGCNLayer")
          .def_readwrite("layer_config", &RGCNLayer::layer_config_)
          .def_readwrite("options", &RGCNLayer::options_)
          .def_readwrite("num_relations", &RGCNLayer::num_relations_)
          .def_readwrite("use_incoming", &RGCNLayer::use_incoming_)
          .def_readwrite("use_outgoing", &RGCNLayer::use_outgoing_)
          .def_readwrite("relation_matrices", &RGCNLayer::relation_matrices_)
          .def_readwrite("inverse_relation_matrices", &RGCNLayer::inverse_relation_matrices_)
          .def_readwrite("self_matrix", &RGCNLayer::self_matrix_)
          .def_readwrite("bias", &RGCNLayer::bias_)
          .def_readwrite("device", &RGCNLayer::device_)
          .def(py::init<shared_ptr<GNNLayerConfig>, int, bool, bool, torch::DeviceType>(),
               py::arg("layer_config"),
               py::arg("num_relations"),
               py::arg("use_incoming"),
               py::arg("use_outgoing"),
               py::arg("device"))
          .def("reset", &RGCNLayer::reset)
          .def("forward", &RGCNLayer::forward,
               py::arg("inputs"),
               py::arg("gnn_graph"),
               py::arg("train") = true);
}