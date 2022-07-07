#include "common/pybind_headers.h"
#include "nn/optim.h"

namespace py = pybind11;

class PyOptimizer : Optimizer {
   public:
    using Optimizer::Optimizer;

    void reset_state() override { PYBIND11_OVERRIDE_PURE_NAME(void, Optimizer, "reset_state", reset_state); }

    void step() override { PYBIND11_OVERRIDE_PURE_NAME(void, Optimizer, "step", step); }
};

void init_optim(py::module &m) {
    py::class_<Optimizer, PyOptimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def_readwrite("num_steps", &Optimizer::num_steps_)
        //        .def_readwrite("state_dict", &Optimizer::state_dict_)
        //        .def_readwrite("param_dict", &Optimizer::param_dict_)
        // TODO need to provide bindings for torch::serialize::InputArchive and torch::serialize::OutputArchive
        //        .def("save", &Optimizer::save, py::arg("output_archive"))
        //        .def("load", &Optimizer::load, py::arg("input_archive"))
        .def("clear_grad", &Optimizer::clear_grad)
        .def("reset_state", &Optimizer::reset_state)
        .def("step", &Optimizer::step);

    py::class_<SGDOptimizer, Optimizer, std::shared_ptr<SGDOptimizer>>(m, "SGDOptimizer")
        .def_readwrite("learning_rate", &SGDOptimizer::learning_rate_)
        .def(py::init<torch::OrderedDict<std::string, torch::Tensor>, float>(), py::arg("param_dict"), py::arg("learning_rate"));

    py::class_<AdagradOptimizer, Optimizer, std::shared_ptr<AdagradOptimizer>>(m, "AdagradOptimizer")
        .def_readwrite("learning_rate", &AdagradOptimizer::learning_rate_)
        .def_readwrite("eps", &AdagradOptimizer::eps_)
        .def_readwrite("lr_decay", &AdagradOptimizer::lr_decay_)
        .def_readwrite("weight_decay", &AdagradOptimizer::weight_decay_)
        .def_readwrite("init_value", &AdagradOptimizer::init_value_)
        .def(py::init<torch::OrderedDict<std::string, torch::Tensor>, std::shared_ptr<AdagradOptions>>(), py::arg("param_dict"), py::arg("options"))
        .def(py::init([](torch::OrderedDict<std::string, torch::Tensor> param_dict, float learning_rate, float eps, float lr_decay, float init_value,
                         float weight_decay) {
                 auto options = std::make_shared<AdagradOptions>();
                 options->learning_rate = learning_rate;
                 options->eps = eps;
                 options->lr_decay = lr_decay;
                 options->init_value = init_value;
                 options->weight_decay = weight_decay;

                 return std::make_shared<AdagradOptimizer>(param_dict, options);
             }),
             py::arg("param_dict"), py::arg("lr") = .1, py::arg("eps") = 1e-10, py::arg("lr_decay") = 0, py::arg("init_value") = 0,
             py::arg("weight_decay") = 0);

    py::class_<AdamOptimizer, Optimizer, std::shared_ptr<AdamOptimizer>>(m, "AdamOptimizer")
        .def_readwrite("learning_rate", &AdamOptimizer::learning_rate_)
        .def_readwrite("eps", &AdamOptimizer::eps_)
        .def_readwrite("beta_1", &AdamOptimizer::beta_1_)
        .def_readwrite("beta_2", &AdamOptimizer::beta_2_)
        .def_readwrite("weight_decay", &AdamOptimizer::weight_decay_)
        .def_readwrite("amsgrad", &AdamOptimizer::amsgrad_)
        .def(py::init<torch::OrderedDict<std::string, torch::Tensor>, std::shared_ptr<AdamOptions>>(), py::arg("param_dict"), py::arg("options"))
        .def(py::init([](torch::OrderedDict<std::string, torch::Tensor> param_dict, float learning_rate, float eps, float beta_1, float beta_2,
                         float weight_decay, bool amsgrad) {
                 auto options = std::make_shared<AdamOptions>();
                 options->learning_rate = learning_rate;
                 options->eps = eps;
                 options->beta_1 = beta_1;
                 options->beta_2 = beta_2;
                 options->weight_decay = weight_decay;
                 options->amsgrad = amsgrad;

                 return std::make_shared<AdamOptimizer>(param_dict, options);
             }),
             py::arg("param_dict"), py::arg("lr") = .1, py::arg("eps") = 1e-8, py::arg("beta_1") = .9, py::arg("beta_2") = .999, py::arg("weight_decay") = 0,
             py::arg("amsgrad") = false);
}