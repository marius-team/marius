//
// Created by Jason Mohoney on 2/14/22.
//

#include "common/pybind_headers.h"
#include "data/samplers/negative.h"

class PyNegativeSampler : NegativeSampler {
   public:
    using NegativeSampler::NegativeSampler;
    using ReturnTensorTuple = std::tuple<torch::Tensor, torch::Tensor>;
    std::tuple<torch::Tensor, torch::Tensor> getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges, bool inverse) override {
        PYBIND11_OVERRIDE_PURE_NAME(ReturnTensorTuple, NegativeSampler, "getNegatives", getNegatives, graph, edges, inverse);
    }
};

void init_neg_samplers(py::module &m) {
    py::class_<NegativeSampler, PyNegativeSampler, std::shared_ptr<NegativeSampler>>(m, "NegativeSampler")
        .def("getNegatives", &NegativeSampler::getNegatives, py::arg("graph"), py::arg("edges") = torch::Tensor(), py::arg("inverse") = false);

    py::class_<CorruptNodeNegativeSampler, NegativeSampler, std::shared_ptr<CorruptNodeNegativeSampler>>(m, "CorruptNodeNegativeSampler")
        .def_readwrite("num_chunks", &CorruptNodeNegativeSampler::num_chunks_)
        .def_readwrite("num_negatives", &CorruptNodeNegativeSampler::num_negatives_)
        .def_readwrite("degree_fraction", &CorruptNodeNegativeSampler::degree_fraction_)
        .def_readwrite("filtered", &CorruptNodeNegativeSampler::filtered_)
        .def(py::init([](int num_chunks, int num_negatives, float degree_fraction, bool filtered, string filter_mode) {
                 auto deg_filter_mode = getLocalFilterMode(filter_mode);
                 return std::make_shared<CorruptNodeNegativeSampler>(num_chunks, num_negatives, degree_fraction, filtered, deg_filter_mode);
             }),
             py::arg("num_chunks") = 1, py::arg("num_negatives") = 500, py::arg("degree_fraction") = 0.0, py::arg("filtered") = false,
             py::arg("local_filter_mode") = "deg");
}
