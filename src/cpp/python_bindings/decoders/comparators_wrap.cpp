//
// Created by Jason Mohoney on 9/30/21.
//

#include "decoders/comparators.h"

#include <torch/extension.h>

class PyComparator : Comparator {
public:
    using Comparator::Comparator;
    using ReturnTensorTuple = std::tuple<torch::Tensor, torch::Tensor>;
    ReturnTensorTuple operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override {
        PYBIND11_OVERRIDE_PURE_NAME(ReturnTensorTuple, Comparator, "__call__", operator(), src, dst, negs); }
};

void init_comparators(py::module &m) {

    py::class_<Comparator, PyComparator>(m, "Comparator")
        .def(py::init<>())
        .def("__call__", &Comparator::operator(), py::arg("src"), py::arg("dst"), py::arg("negs"));

    py::class_<L2Compare, Comparator>(m, "L2Compare")
        .def(py::init<>());

    py::class_<CosineCompare, Comparator>(m, "CosineCompare")
        .def(py::init<>());

    py::class_<DotCompare, Comparator>(m, "DotCompare")
        .def(py::init<>());
}