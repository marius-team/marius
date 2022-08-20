//
// Created by Jason Mohoney on 9/30/21.
//

#include "layers/graph_sage_layer.h"

#include <torch/extension.h>

void init_graph_sage_layer(py::module &m) {

    torch::python::bind_module<GraphSageLayer>(m, "GraphSageLayer")
            .def_readwrite("layer_config", &GraphSageLayer::layer_config_)
            .def_readwrite("options", &GraphSageLayer::options_)
            .def_readwrite("use_incoming", &GraphSageLayer::use_incoming_)
            .def_readwrite("use_outgoing", &GraphSageLayer::use_outgoing_)
            .def_readwrite("w1", &GraphSageLayer::w1_)
            .def_readwrite("w2", &GraphSageLayer::w2_)
            .def_readwrite("bias", &GraphSageLayer::bias_)
            .def_readwrite("device", &GraphSageLayer::device_)

            .def(py::init([](int input_dim,
                             int output_dim,
                             string aggregator,
                             string init,
                             shared_ptr<InitOptions> init_options,
                             bool use_incoming,
                             bool use_outgoing,
                             bool use_bias,
                             string bias_init,
                             shared_ptr<InitOptions> bias_init_options,
                             py::object py_device) {

                     auto config = std::make_shared<GNNLayerConfig>();

                     auto options = std::make_shared<GraphSageLayerOptions>();
                     options->input_dim = input_dim;
                     options->output_dim = output_dim;
                     options->aggregator = getGraphSageAggregator(aggregator);
                     config->options = options;

                     config->init = std::make_shared<InitConfig>();
                     config->init->type = getInitDistribution(init);
                     config->init->options = init_options;

                     config->bias = use_bias;

                     if (config->bias) {
                         config->bias_init = std::make_shared<InitConfig>();
                         config->bias_init->type = getInitDistribution(bias_init);
                         config->bias_init->options = bias_init_options;
                     }

                     return std::unique_ptr<GraphSageLayer>(new GraphSageLayer(config, use_incoming, use_outgoing, torch::python::detail::py_object_to_device(py_device)));
                 }),
                 py::arg("input_dim"),
                 py::arg("output_dim"),
                 py::arg("aggregator") = "mean",
                 py::arg("init") = "glorot_uniform",
                 py::arg("init_options") = nullptr,
                 py::arg("use_incoming") = true,
                 py::arg("use_outgoing") = true,
                 py::arg("use_bias") = true,
                 py::arg("bias_init") = "ones",
                 py::arg("bias_init_options") = nullptr,
                 py::arg("device") = torch::kCPU)
            .def("reset", &GraphSageLayer::reset)
            .def("forward", &GraphSageLayer::forward,
                 py::arg("inputs"),
                 py::arg("gnn_graph"),
                 py::arg("train") = true);
}