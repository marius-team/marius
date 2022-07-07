#include "common/pybind_headers.h"
#include "pipeline/graph_encoder.h"

namespace py = pybind11;

// Trampoline class
class PyGraphEncoder : GraphEncoder {
   public:
    using GraphEncoder::GraphEncoder;
    void encode(bool separate_layers) override { PYBIND11_OVERRIDE_PURE(void, GraphEncoder, encode, separate_layers); }
};

void init_graph_encoder(py::module &m) {
    py::class_<GraphEncoder, PyGraphEncoder, std::shared_ptr<GraphEncoder>>(m, "GraphEncoder")
        .def_readwrite("dataloader", &GraphEncoder::dataloader_)
        .def_readwrite("progress_reporter", &GraphEncoder::progress_reporter_)
        .def("encode", &GraphEncoder::encode, py::arg("separate_layers") = false);

    py::class_<SynchronousGraphEncoder, GraphEncoder, std::shared_ptr<SynchronousGraphEncoder>>(m, "SynchronousEncoder")
        .def(py::init<shared_ptr<DataLoader>, std::shared_ptr<Model>>(), py::arg("dataloader"), py::arg("model"));

    py::class_<PipelineGraphEncoder, GraphEncoder, std::shared_ptr<PipelineGraphEncoder>>(m, "PipelineEncoder")
        .def(py::init<shared_ptr<DataLoader>, std::shared_ptr<Model>, std::shared_ptr<PipelineConfig>>(), py::arg("dataloader"), py::arg("model"),
             py::arg("pipeline_config"));
}
