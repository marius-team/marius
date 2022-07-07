//
// Created by Jason Mohoney on 3/23/21.
//

#include "common/pybind_headers.h"
#include "configuration/config.h"
#include "configuration/util.h"
#include "nn/model.h"

class PyModel : Model {
   public:
    using Model::Model;
};

void init_model(py::module &m) {
    py::class_<Model, PyModel, torch::nn::Module, shared_ptr<Model>>(m, "Model", py::dynamic_attr())
        .def_readwrite("encoder", &Model::encoder_)
        .def_readwrite("decoder", &Model::decoder_)
        .def_readwrite("optimizers", &Model::optimizers_)
        .def_readwrite("loss_function", &Model::loss_function_)
        .def_readwrite("reporter", &Model::reporter_)
        .def_readwrite("device", &Model::device_)
        .def_readwrite("learning_task", &Model::learning_task_)
        .def_readwrite("sparse_lr", &Model::sparse_lr_)
        .def_readwrite("device_models", &Model::device_models_)
        .def(py::init<shared_ptr<GeneralEncoder>, shared_ptr<Decoder>, shared_ptr<LossFunction>, shared_ptr<Reporter>>())
        .def(py::init([](shared_ptr<GeneralEncoder> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Reporter> reporter,
                         float sparse_lr) {
                 auto model = std::make_shared<Model>(encoder, decoder, loss, reporter);
                 model->sparse_lr_ = sparse_lr;
                 return model;
             }),
             py::arg("encoder"), py::arg("decoder"), py::arg("loss") = nullptr, py::arg("reporter") = nullptr, py::arg("sparse_lr") = .1)

        .def("forward_nc", &Model::forward_nc, py::arg("node_embeddings"), py::arg("node_features"), py::arg("dense_graph"), py::arg("train"),
             py::call_guard<py::gil_scoped_release>())
        .def("forward_lp", &Model::forward_lp, py::arg("batch"), py::arg("train"), py::call_guard<py::gil_scoped_release>())
        .def("train_batch", &Model::train_batch, py::arg("batch"), py::arg("call_step") = true, py::call_guard<py::gil_scoped_release>())
        .def("evaluate_batch", &Model::evaluate_batch, py::arg("batch"), py::call_guard<py::gil_scoped_release>())
        .def("clear_grad", &Model::clear_grad)
        .def("clear_grad_all", &Model::clear_grad_all)
        .def("step", &Model::step)
        .def("step_all", &Model::step_all)
        .def("save", &Model::save, py::arg("directory"))
        .def("load", &Model::load, py::arg("directory"), py::arg("train"))
        .def("broadcast", &Model::broadcast, py::arg("devices"))
        .def("all_reduce", &Model::all_reduce);

    m.def(
        "initModelFromConfig",
        [](pyobj python_config, pybind11::list devices_pylist, int num_relations, bool train) {
            std::vector<torch::Device> devices = {};

            for (auto py_id : devices_pylist) {
                pyobj id_object = pybind11::reinterpret_borrow<pyobj>(py_id);
                devices.emplace_back(torch::python::detail::py_object_to_device(id_object));
            }

            shared_ptr<ModelConfig> model_config = initModelConfig(python_config);

            return initModelFromConfig(model_config, devices, num_relations, train);
        },
        py::arg("model_config"), py::arg("devices"), py::arg("num_relations"), py::arg("train"), py::return_value_policy::move);

    m.def(
        "load_from_file",
        [](string config_path, bool train) {
            auto config = loadConfig(config_path, false);
            auto devices = devices_from_config(config->storage);
            auto model = initModelFromConfig(config->model, devices, config->storage->dataset->num_relations, train);
            model->load(config->storage->model_dir, train);
            return model;
        },
        py::arg("config_path"), py::arg("train"), py::return_value_policy::move);
}