#include "torch/extension.h"

#include "trainer.h"

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
            .def_readwrite("dataloader", &Trainer::dataloader_)
            .def_readwrite("progress_reporter", &Trainer::progress_reporter_)
            .def_readwrite("learning_task", &Trainer::learning_task_)
            .def("train", &Trainer::train, py::arg("num_epochs") = 1, py::call_guard<py::gil_scoped_release>());

    py::class_<SynchronousTrainer, Trainer>(m, "SynchronousTrainer")
            .def(py::init([](DataLoader *dataloader, std::shared_ptr<torch::nn::Module> model, int logs_per_epoch) {
                std::shared_ptr<Model> casted_model = std::dynamic_pointer_cast<Model>(model);
                return new SynchronousTrainer(dataloader, casted_model, logs_per_epoch);
            }), py::arg("sampler"), py::arg("model"), py::arg("logs_per_epoch"), py::return_value_policy::reference_internal);

    py::class_<PipelineTrainer, Trainer>(m, "PipelineTrainer")
            .def(py::init([](DataLoader *dataloader, std::shared_ptr<torch::nn::Module> model, shared_ptr<PipelineConfig> config, int logs_per_epoch) {
                return new PipelineTrainer(dataloader, std::dynamic_pointer_cast<Model>(model), config, logs_per_epoch);
            }), py::arg("sampler"), py::arg("model"), py::arg("pipeline_config"), py::arg("logs_per_epoch"), py::return_value_policy::reference_internal);
}