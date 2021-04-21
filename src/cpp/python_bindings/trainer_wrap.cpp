#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <trainer.h>

namespace py = pybind11;

// Trampoline class
class PyTrainer : Trainer {
  public:
    using Trainer::Trainer;
    void train(int num_epochs = 1) override { PYBIND11_OVERRIDE_PURE(void, Trainer, train, num_epochs); }
};

void init_trainer(py::module &m) {
    py::class_<Trainer, PyTrainer>(m, "Trainer")
        .def(py::init<>())
        .def_readwrite("data_set", &Trainer::data_set_)
        .def("train", &Trainer::train, py::arg("num_epochs") = 1, py::call_guard<py::gil_scoped_release>());

    py::class_<PipelineTrainer, Trainer>(m, "PipelineTrainer")
        .def(py::init<DataSet *, Model *>(), py::arg("data_set"), py::arg("model"));

    py::class_<SynchronousTrainer, Trainer>(m, "SynchronousTrainer")
        .def(py::init<DataSet *, Model *>(), py::arg("data_set"), py::arg("model"));
}
