#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <loss.h>

namespace py = pybind11;

class PyLossFunction : LossFunction {
  public:
    using LossFunction::LossFunction;
    torch::Tensor operator()(torch::Tensor pos_scores, torch::Tensor neg_scores) override { 
      PYBIND11_OVERRIDE_NAME(torch::Tensor, LossFunction, "__call__", operator(), pos_scores, neg_scores); }
};

void init_loss(py::module &m) {
  py::class_<LossFunction, PyLossFunction>(m, "LossFunction")
    .def("__call__", &LossFunction::operator(), py::arg("pos_scores"), py::arg("neg_scores"));
  py::class_<SoftMax, LossFunction>(m, "SoftMax")
    .def(py::init<>());
  py::class_<RankingLoss, LossFunction>(m, "RankingLoss")
    .def(py::init<float>(), py::arg("margin"));
  py::class_<BCEAfterSigmoidLoss, LossFunction>(m, "BCEAfterSigmoidLoss")
    .def(py::init<>());
  py::class_<BCEWithLogitsLoss, LossFunction>(m, "BCEWithLogitsLoss")
    .def(py::init<>());
  py::class_<MSELoss, LossFunction>(m, "MSELoss")
    .def(py::init<>());
  py::class_<SoftPlusLoss, LossFunction>(m, "SoftPlusLoss")
    .def(py::init<>());
}