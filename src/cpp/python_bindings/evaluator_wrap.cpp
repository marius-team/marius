#include <torch/extension.h>

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
        .def_readwrite("dataloader", &Evaluator::dataloader_)
        .def("evaluate", &Evaluator::evaluate, py::arg("validation"));

    py::class_<SynchronousEvaluator, Evaluator>(m, "SynchronousEvaluator")
            .def(py::init([](DataLoader *dataloader, std::shared_ptr<torch::nn::Module> model) {
                std::shared_ptr<Model> casted_model = std::dynamic_pointer_cast<Model>(model);
                return new SynchronousEvaluator(dataloader, casted_model);
            }), py::arg("sampler"), py::arg("model"), py::return_value_policy::reference_internal);

    py::class_<PipelineEvaluator, Evaluator>(m, "PipelineEvaluator")
            .def(py::init([](DataLoader *dataloader, std::shared_ptr<torch::nn::Module> model, shared_ptr<PipelineConfig> config) {
                return new PipelineEvaluator(dataloader, std::dynamic_pointer_cast<Model>(model), config);
            }), py::arg("sampler"), py::arg("model"), py::arg("pipeline_config"), py::return_value_policy::reference_internal);
}
