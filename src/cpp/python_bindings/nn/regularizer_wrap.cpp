#include "common/pybind_headers.h"
#include "nn/regularizer.h"

namespace py = pybind11;

class PyRegularizer : Regularizer {
   public:
    using Regularizer::Regularizer;
    torch::Tensor operator()(torch::Tensor src_nodes_embs, torch::Tensor dst_node_embs) override {
        PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, Regularizer, "__call__", operator(), src_nodes_embs, dst_node_embs);
    }
};

void init_regularizer(py::module &m) {
    py::class_<Regularizer, PyRegularizer>(m, "Regularizer")
        .def(py::init<>())
        .def("__call__", &Regularizer::operator(), py::arg("src_nodes_embs"), py::arg("dst_node_embs"));

    py::class_<NormRegularizer, Regularizer>(m, "NormRegularizer").def(py::init<int, float>(), py::arg("norm"), py::arg("coefficient"));
}