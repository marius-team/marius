//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/layers/reduction/reduction_layer.h"

namespace py = pybind11;

class PyLayer : Layer {
   public:
    using Layer::Layer;
};

void init_layer(py::module &m) {
    py::class_<Layer, PyLayer, torch::nn::Module, std::shared_ptr<Layer>>(m, "Layer")
        .def_readwrite("config", &Layer::config_)
        .def_readwrite("device", &Layer::device_)
        .def_readwrite("bias", &Layer::bias_)
        .def("post_hook", &ReductionLayer::post_hook, py::arg("inputs"))
        .def("init_bias", &ReductionLayer::init_bias);
}