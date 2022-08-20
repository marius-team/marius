#include "featurizers/featurizer.h"
#include "torch/extension.h"

namespace py = pybind11;

// Trampoline class
// class PyFeaturizer : Featurizer {
//   public:
//     using Featurizer::Featurizer;
//     Embeddings operator()(Features node_features, Embeddings node_embeddings) override {
//         PYBIND11_OVERRIDE_PURE_NAME(Embeddings, Featurizer, "__call__", operator(), node_features, node_embeddings); }
// };

void init_featurizer(py::module &m) {
    // torch::python::bind_module<Featurizer>(m, "Featurizer")
    //   .def(py::init<>())
    //   .def("__call__", &Featurizer::operator(), py::arg("node_features"), py::arg("node_embeddings"));

    // torch::python::bind_module<CatFeaturizer>(m, "CatFeaturizer")
    //   .def(py::init<int, float>(), py::arg("norm"), py::arg("coefficient"));
}