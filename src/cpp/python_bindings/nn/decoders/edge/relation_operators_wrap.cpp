//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/decoders/edge/relation_operators.h"

// Trampoline classes
class PyRelationOperator : RelationOperator {
   public:
    using RelationOperator::RelationOperator;
    torch::Tensor operator()(const torch::Tensor &embs, const torch::Tensor &rels) override {
        PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, RelationOperator, "__call__", operator(), embs, rels);
    }
};

void init_relation_operators(py::module &m) {
    py::class_<RelationOperator, PyRelationOperator, std::shared_ptr<RelationOperator>>(m, "RelationOperator")
        .def(py::init<>())
        .def("__call__", &RelationOperator::operator(), py::arg("embs"), py::arg("rels"));

    py::class_<HadamardOperator, RelationOperator, std::shared_ptr<HadamardOperator>>(m, "HadamardOperator").def(py::init<>());

    py::class_<ComplexHadamardOperator, RelationOperator, std::shared_ptr<ComplexHadamardOperator>>(m, "ComplexHadamardOperator").def(py::init<>());

    py::class_<TranslationOperator, RelationOperator, std::shared_ptr<TranslationOperator>>(m, "TranslationOperator").def(py::init<>());

    py::class_<NoOp, RelationOperator, std::shared_ptr<NoOp>>(m, "NoOp").def(py::init<>());
}
