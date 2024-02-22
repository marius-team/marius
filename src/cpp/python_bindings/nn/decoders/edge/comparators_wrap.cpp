//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/decoders/edge/comparators.h"

class PyComparator : Comparator {
   public:
    using Comparator::Comparator;
    torch::Tensor operator()(torch::Tensor src, torch::Tensor dst) override {
        PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, Comparator, "__call__", operator(), src, dst);
    }
};

void init_comparators(py::module &m) {
    py::class_<Comparator, PyComparator, shared_ptr<Comparator>>(m, "Comparator").def("__call__", &Comparator::operator(), py::arg("src"), py::arg("dst"));

    py::class_<L2Compare, Comparator, shared_ptr<L2Compare>>(m, "L2Compare").def(py::init<>());

    py::class_<CosineCompare, Comparator, std::shared_ptr<CosineCompare>>(m, "CosineCompare").def(py::init<>());

    py::class_<DotCompare, Comparator, std::shared_ptr<DotCompare>>(m, "DotCompare").def(py::init<>());
}