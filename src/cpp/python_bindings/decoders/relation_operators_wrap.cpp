//
// Created by Jason Mohoney on 9/30/21.
//

#include "decoders/relation_operators.h"

#include <torch/extension.h>

// Trampoline classes
class PyRelationOperator : RelationOperator {
public:
    using RelationOperator::RelationOperator;
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override {
        PYBIND11_OVERRIDE_PURE_NAME(Embeddings, RelationOperator, "__call__", operator(), embs, rels); }
};

void init_relation_operators(py::module &m) {

    py::class_<RelationOperator, PyRelationOperator>(m, "RelationOperator")
        .def(py::init<>())
        .def("__call__", &RelationOperator::operator(), py::arg("embs"), py::arg("rels"));

    py::class_<HadamardOperator, RelationOperator>(m, "HadamardOperator")
        .def(py::init<>());

    py::class_<ComplexHadamardOperator, RelationOperator>(m, "ComplexHadamardOperator")
        .def(py::init<>());

    py::class_<TranslationOperator, RelationOperator>(m, "TranslationOperator")
        .def(py::init<>());

    py::class_<NoOp, RelationOperator>(m, "NoOp")
        .def(py::init<>());
}
