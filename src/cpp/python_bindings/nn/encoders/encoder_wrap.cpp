
#include "common/pybind_headers.h"
#include "nn/encoders/encoder.h"

void init_encoder(py::module &m) {
    py::class_<GeneralEncoder, torch::nn::Module, std::shared_ptr<GeneralEncoder>>(m, "GeneralEncoder")
        .def_readwrite("encoder_config", &GeneralEncoder::encoder_config_)
        .def_readwrite("num_relations", &GeneralEncoder::num_relations_)
        .def_readwrite("device", &GeneralEncoder::device_)
        .def_readwrite("layers", &GeneralEncoder::layers_)
        .def(py::init<shared_ptr<EncoderConfig>, torch::Device, int>(), py::arg("encoder_config"), py::arg("device"), py::arg("num_relations") = 1)
        .def(py::init<std::vector<std::vector<shared_ptr<Layer>>>>(), py::arg("layers"))
        .def("forward", &GeneralEncoder::forward, py::arg("embeddings"), py::arg("features"), py::arg("dense_graph"), py::arg("train") = true)
        .def("reset", &GeneralEncoder::reset);
}