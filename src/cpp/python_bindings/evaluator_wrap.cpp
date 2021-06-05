#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "evaluator.h"

namespace py = pybind11;

// Trampoline class
class PyEvaluator : Evaluator {
  public:
    using Evaluator::Evaluator;
    void evaluate(bool validation) override { PYBIND11_OVERRIDE_PURE(void, Evaluator, evaluate, validation); }
};

void init_evaluator(py::module &m) {
	py::class_<Evaluator, PyEvaluator>(m, "Evaluator")
         .def(py::init<>())
         .def_readwrite("data_set", &Evaluator::data_set_)
         .def("evaluate", &Evaluator::evaluate, py::arg("validation"));

    py::class_<PipelineEvaluator, Evaluator>(m, "PipelineEvaluator")
        .def(py::init<DataSet *, Model *>(), py::arg("data_set"), py::arg("model"));

    py::class_<SynchronousEvaluator, Evaluator>(m, "SynchronousEvaluator")
        .def(py::init<DataSet *, Model *>(), py::arg("data_set"), py::arg("model"));
}
