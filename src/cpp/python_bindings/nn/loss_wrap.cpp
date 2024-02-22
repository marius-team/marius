#include "common/pybind_headers.h"
#include "nn/loss.h"

namespace py = pybind11;

class PyLossFunction : LossFunction {
   public:
    using LossFunction::LossFunction;
    torch::Tensor operator()(torch::Tensor y_pred, torch::Tensor targets, bool scores) override {
        PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, LossFunction, "__call__", operator(), y_pred, targets, scores);
    }
};

void init_loss(py::module &m) {
    py::class_<LossFunction, PyLossFunction, shared_ptr<LossFunction>>(m, "LossFunction")
        .def(py::init<>())
        .def("__call__", &LossFunction::operator(), py::arg("y_pred"), py::arg("targets"), py::arg("scores"));

    py::class_<SoftmaxCrossEntropy, LossFunction, shared_ptr<SoftmaxCrossEntropy>>(m, "SoftmaxCrossEntropy")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<SoftmaxCrossEntropy>(options);
             }),
             py::arg("reduction") = "sum");

    py::class_<RankingLoss, LossFunction, shared_ptr<RankingLoss>>(m, "RankingLoss")
        .def(py::init([](string reduction, float margin) {
                 auto options = std::make_shared<RankingLossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 options->margin = margin;
                 return std::make_shared<RankingLoss>(options);
             }),
             py::arg("reduction") = "sum", py::arg("margin") = 1.0);

    py::class_<CrossEntropyLoss, LossFunction, shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<CrossEntropyLoss>(options);
             }),
             py::arg("reduction") = "sum");

    py::class_<BCEAfterSigmoidLoss, LossFunction, shared_ptr<BCEAfterSigmoidLoss>>(m, "BCEAfterSigmoidLoss")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<BCEAfterSigmoidLoss>(options);
             }),
             py::arg("reduction") = "sum");

    py::class_<BCEWithLogitsLoss, LossFunction, shared_ptr<BCEWithLogitsLoss>>(m, "BCEWithLogitsLoss")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<BCEWithLogitsLoss>(options);
             }),
             py::arg("reduction") = "sum");

    py::class_<MSELoss, LossFunction, shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<MSELoss>(options);
             }),
             py::arg("reduction") = "sum");

    py::class_<SoftPlusLoss, LossFunction, shared_ptr<SoftPlusLoss>>(m, "SoftPlusLoss")
        .def(py::init([](string reduction) {
                 auto options = std::make_shared<LossOptions>();
                 options->loss_reduction = getLossReduction(reduction);
                 return std::make_shared<SoftPlusLoss>(options);
             }),
             py::arg("reduction") = "sum");

    m.def("getLossFunction", &getLossFunction, py::arg("config"));
}