#include "common/pybind_headers.h"
#include "nn/layers/embedding/embedding.h"

void init_embedding_layer(py::module &m) {
    py::class_<EmbeddingLayer, Layer, std::shared_ptr<EmbeddingLayer>>(m, "EmbeddingLayer")
        .def_readwrite("offset", &EmbeddingLayer::offset_)
        .def(py::init<shared_ptr<LayerConfig>, torch::Device, int>(), py::arg("layer_config"), py::arg("device"), py::arg("offset") = 0)
        .def(py::init([](int dimension, torch::Device device, InitConfig init, bool bias, InitConfig bias_init, string activation, int offset) {
                 auto layer_config = std::make_shared<LayerConfig>();
                 layer_config->input_dim = -1;
                 layer_config->output_dim = dimension;
                 layer_config->type = LayerType::EMBEDDING;
                 layer_config->init = std::make_shared<InitConfig>(init);
                 layer_config->bias = bias;
                 layer_config->bias_init = std::make_shared<InitConfig>(bias_init);
                 layer_config->optimizer = nullptr;
                 layer_config->activation = getActivationFunction(activation);

                 return std::make_shared<EmbeddingLayer>(layer_config, device, offset);
             }),
             py::arg("dimension"), py::arg("device"), py::arg("init") = InitConfig(InitDistribution::GLOROT_UNIFORM, nullptr), py::arg("bias") = false,
             py::arg("bias_init") = InitConfig(InitDistribution::ZEROS, nullptr), py::arg("activation") = "none", py::arg("offset") = 0)
        .def("init_embeddings", &EmbeddingLayer::init_embeddings, py::arg("num_nodes"))
        .def("forward", &EmbeddingLayer::forward, py::arg("input"))
        .def("reset", &EmbeddingLayer::reset);
}