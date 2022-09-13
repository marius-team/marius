#include "common/pybind_headers.h"
#include "pipeline/trainer.h"

namespace py = pybind11;

// Trampoline class
class PyTrainer : Trainer {
   public:
    using Trainer::Trainer;
    void train(int num_epochs = 1) override { PYBIND11_OVERRIDE_PURE(void, Trainer, train, num_epochs); }
};

void init_trainer(py::module &m) {
    py::class_<Trainer, PyTrainer, shared_ptr<Trainer>>(m, "Trainer")
        .def(py::init<>())
        .def_readwrite("dataloader", &Trainer::dataloader_)
        .def_readwrite("progress_reporter", &Trainer::progress_reporter_)
        .def_readwrite("learning_task", &Trainer::learning_task_)
        .def("train", &Trainer::train, py::arg("num_epochs") = 1);

    py::class_<SynchronousTrainer, Trainer, shared_ptr<SynchronousTrainer>>(m, "SynchronousTrainer")
        .def(py::init<shared_ptr<DataLoader>, shared_ptr<Model>, int>(), py::arg("dataloader"), py::arg("model"), py::arg("logs_per_epoch") = 10);

    py::class_<PipelineTrainer, Trainer, shared_ptr<PipelineTrainer>>(m, "PipelineTrainer")
        .def(py::init<shared_ptr<DataLoader>, shared_ptr<Model>, shared_ptr<PipelineConfig>, int>(), py::arg("dataloader"), py::arg("model"),
             py::arg("pipeline_config"), py::arg("logs_per_epoch") = 10);
}