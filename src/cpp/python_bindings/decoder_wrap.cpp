#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "decoder.h"

namespace py = pybind11;

// Trampoline classes
class PyRelationOperator : RelationOperator {
  public:
    using RelationOperator::RelationOperator;
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override { 
      PYBIND11_OVERRIDE_NAME(Embeddings, RelationOperator, "__call__", operator(), embs, rels); }
};

// this guy doesn't like the tuple of tensors
// class PyComparator : Comparator {
//   public:
//     using Comparator::Comparator;
//     tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override { 
//       PYBIND11_OVERRIDE_NAME(tuple<torch::Tensor, torch::Tensor>, Comparator, "__call__", operator(), src, dst, negs); }
// };

class PyDecoder : Decoder {
  public:
    using Decoder::Decoder;
    void forward(Batch *batch, bool train) override { 
      PYBIND11_OVERRIDE_PURE(void, Decoder, forward, batch, train); }
};

void init_decoder(py::module &m) {

  py::class_<RelationOperator, PyRelationOperator>(m, "RelationOperator")
    .def("__call__", &RelationOperator::operator(), py::arg("embs"), py::arg("rels"));
  py::class_<HadamardOperator, RelationOperator>(m, "HadamardOperator");
  py::class_<ComplexHadamardOperator, RelationOperator>(m, "ComplexHadamardOperator");
  py::class_<TranslationOperator, RelationOperator>(m, "TranslationOperator");
  py::class_<NoOp, RelationOperator>(m, "NoOp");
  
  // py::class_<Comparator, PyComparator>(m, "Comparator")
  //   .def("__call__", &Comparator::operator(), py::arg("src"), py::arg("dst"), py::arg("negs"));
  // py::class_<CosineCompare, Comparator>(m, "CosineCompare")
  //   .def(py::init<>());
  // py::class_<DotCompare, Comparator>(m, "DotCompare")
  //   .def(py::init<>());

  py::class_<Decoder, PyDecoder>(m, "Decoder")
    .def(py::init<>())
    .def("forward", &Decoder::forward, py::arg("batch"), py::arg("train"));
  py::class_<LinkPredictionDecoder, Decoder>(m, "LinkPredictionDecoder")
    .def(py::init<>())
    .def(py::init<Comparator *, RelationOperator *, LossFunction *>());
  py::class_<DistMult, LinkPredictionDecoder>(m, "DistMult")
    .def(py::init<>());
  py::class_<TransE, LinkPredictionDecoder>(m, "TransE")
    .def(py::init<>());
  py::class_<ComplEx, LinkPredictionDecoder>(m, "ComplEx")
    .def(py::init<>());
  py::class_<NodeClassificationDecoder, Decoder>(m, "NodeClassificationDecoder")
    .def(py::init<>());
  py::class_<RelationClassificationDecoder, Decoder>(m, "RelationClassificationDecoder");
}
