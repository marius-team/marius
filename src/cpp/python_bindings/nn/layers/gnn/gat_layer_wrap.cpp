//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/layers/gnn/gat_layer.h"

void init_gat_layer(py::module &m) {
    py::class_<GATLayer, GNNLayer, shared_ptr<GATLayer>>(m, "GATLayer")
        .def_readwrite("options", &GATLayer::options_)
        .def_readwrite("head_dim", &GATLayer::head_dim_)
        .def_readwrite("input_dropout", &GATLayer::input_dropout_)
        .def_readwrite("attention_dropout", &GATLayer::attention_dropout_)
        .def_readwrite("weight_matrices", &GATLayer::weight_matrices_)
        .def_readwrite("a_l", &GATLayer::a_l_)
        .def_readwrite("a_r", &GATLayer::a_r_)
        .def(py::init<shared_ptr<LayerConfig>, torch::Device>(), py::arg("layer_config"), py::arg("device"))
        .def(py::init([](int input_dim, int output_dim, std::optional<torch::Device> device, int num_heads, bool average_heads, float input_dropout,
                         float attention_dropout, float negative_slope, InitConfig init, bool bias, InitConfig bias_init, string activation) {
                 auto layer_config = std::make_shared<LayerConfig>();
                 layer_config->input_dim = input_dim;
                 layer_config->output_dim = output_dim;
                 layer_config->type = LayerType::GNN;

                 auto layer_options = std::make_shared<GATLayerOptions>();
                 layer_options->input_dropout = input_dropout;
                 layer_options->attention_dropout = attention_dropout;
                 layer_options->num_heads = num_heads;
                 layer_options->negative_slope = negative_slope;
                 layer_options->average_heads = average_heads;
                 layer_config->options = layer_options;

                 layer_config->init = std::make_shared<InitConfig>(init);
                 layer_config->bias = bias;
                 layer_config->bias_init = std::make_shared<InitConfig>(bias_init);
                 layer_config->optimizer = nullptr;
                 layer_config->activation = getActivationFunction(activation);

                 torch::Device torch_device = torch::kCPU;
                 if (device.has_value()) {
                     torch_device = device.value();
                 }

                 return std::make_shared<GATLayer>(layer_config, torch_device);
             }),
             py::arg("input_dim"), py::arg("output_dim"), py::arg("device") = py::none(), py::arg("num_heads") = 10, py::arg("average_heads") = false,
             py::arg("input_dropout") = 0.0, py::arg("attention_dropout") = 0.0, py::arg("negative_slope") = .2,
             py::arg("init") = InitConfig(InitDistribution::GLOROT_UNIFORM, nullptr), py::arg("bias") = false,
             py::arg("bias_init") = InitConfig(InitDistribution::ZEROS, nullptr), py::arg("activation") = "none")
        .def("reset", &GATLayer::reset)
        .def("forward", &GATLayer::forward, py::arg("inputs"), py::arg("dense_graph"), py::arg("train") = true);
}
