//
// Created by Jason Mohoney on 2/15/22.
//

#include "common/pybind_headers.h"
#include "nn/layers/feature/feature.h"

void init_feature_layer(py::module &m) {
    py::class_<FeatureLayer, Layer, std::shared_ptr<FeatureLayer>>(m, "FeatureLayer")
        .def_readwrite("offset", &FeatureLayer::offset_)
        .def(py::init<shared_ptr<LayerConfig>, torch::Device, int>(), py::arg("layer_config"), py::arg("device"), py::arg("offset") = 0)
        .def(py::init([](int dimension, torch::Device device, bool bias, InitConfig bias_init, string activation, int offset) {
                 auto layer_config = std::make_shared<LayerConfig>();
                 layer_config->input_dim = -1;
                 layer_config->output_dim = dimension;
                 layer_config->type = LayerType::FEATURE;
                 layer_config->init = nullptr;
                 layer_config->bias = bias;
                 layer_config->bias_init = std::make_shared<InitConfig>(bias_init);
                 layer_config->optimizer = nullptr;
                 layer_config->activation = getActivationFunction(activation);

                 return std::make_shared<FeatureLayer>(layer_config, device, offset);
             }),
             py::arg("dimension"), py::arg("device"), py::arg("bias") = false, py::arg("bias_init") = InitConfig(InitDistribution::ZEROS, nullptr),
             py::arg("activation") = "none", py::arg("offset") = 0)
        .def("forward", &FeatureLayer::forward, py::arg("input"))
        .def("reset", &FeatureLayer::reset);
}