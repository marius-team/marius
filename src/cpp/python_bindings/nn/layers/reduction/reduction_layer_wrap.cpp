//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/layers/reduction/reduction_layer.h"

class PyReductionLayer : ReductionLayer {
   public:
    using ReductionLayer::ReductionLayer;
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override { PYBIND11_OVERRIDE_PURE(torch::Tensor, ReductionLayer, forward, inputs); }
};

void init_reduction_layer(py::module &m) {
    py::class_<ReductionLayer, PyReductionLayer, Layer, std::shared_ptr<ReductionLayer>>(m, "ReductionLayer")
        .def("forward", &ReductionLayer::forward, py::arg("inputs"));
}