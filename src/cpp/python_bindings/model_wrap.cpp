//
// Created by Jason Mohoney on 3/23/21.
//

#include "torch/extension.h"

#include "model.h"
#include "configuration/config.h"

namespace py = pybind11;

void init_model(py::module &m) {

    py::class_<torch::nn::Module, std::shared_ptr<torch::nn::Module>>(m, "torch::nn::Module");
    py::class_<Model, torch::nn::Module, std::shared_ptr<Model>>(m, "Model")
            .def_readwrite("featurizer", &Model::featurizer_)
            .def_readwrite("featurizer_optimizer", &Model::featurizer_optimizer_)
            .def_readwrite("encoder", &Model::encoder_)
            .def_readwrite("encoder_optimizer", &Model::encoder_optimizer_)
            .def_readwrite("decoder", &Model::decoder_)
            .def_readwrite("decoder_optimizer", &Model::decoder_optimizer_)
            .def_readwrite("loss_function", &Model::loss_function_)
            .def_readwrite("regularizer", &Model::regularizer_)
            .def_readwrite("reporter", &Model::reporter_)
            .def_readwrite("current_device", &Model::current_device_)
            .def_readwrite("devices", &Model::devices_)
            .def_readwrite("learning_task", &Model::learning_task_)
            .def_readwrite("model_config", &Model::model_config_)
            .def_readwrite("has_embeddings", &Model::has_embeddings_)
            .def_readwrite("has_features", &Model::has_features_)
            .def_readwrite("reinitialize", &Model::reinitialize_)
            .def_readwrite("is_train", &Model::train_)
            .def_readwrite("filtered_eval", &Model::filtered_eval_)
            .def_readwrite("device_models", &Model::device_models_)
            .def(py::init<>())
            .def(py::init<shared_ptr<ModelConfig>, torch::Device>(), py::arg("model_config"), py::arg("device"))
            .def("train_batch", static_cast<void (Model::*)(Batch *)>(&Model::train_batch), py::arg("batch"), py::call_guard<py::gil_scoped_release>())
            .def("train_batch", static_cast<void (Model::*)(std::vector<Batch *>)>(&Model::train_batch), py::arg("sub_batches"),
                 py::call_guard<py::gil_scoped_release>())
            .def("evaluate", static_cast<void (Model::*)(Batch *, bool)>(&Model::evaluate), py::arg("batch"), py::arg("filtered_eval"),
                 py::call_guard<py::gil_scoped_release>())
            .def("evaluate", static_cast<void (Model::*)(std::vector<Batch *>, bool)>(&Model::evaluate), py::arg("sub_batches"), py::arg("filtered_eval"),
                 py::call_guard<py::gil_scoped_release>())
            .def("zero_grad", &Model::zero_grad)
            .def("step", &Model::step)
            .def("save", &Model::save, py::arg("directory"))
            .def("load", &Model::load, py::arg("directory"))
            .def("clone_to_device", &Model::clone_to_device, py::arg("device"))
            .def("broadcast", &Model::broadcast, py::arg("devices"))
            .def("allReduce", &Model::allReduce);

//    torch::python::bind_module<NodeClassificationModel, Model>(m, "NodeClassificationModel")
//        .def_readwrite("featurizer", &Model::featurizer_)
//        .def_readwrite("featurizer_optimizer", &Model::featurizer_optimizer_)
//        .def_readwrite("encoder", &Model::encoder_)
//        .def_readwrite("encoder_optimizer", &Model::encoder_optimizer_)
//        .def_readwrite("decoder", &Model::decoder_)
//        .def_readwrite("decoder_optimizer", &Model::decoder_optimizer_)
//        .def_readwrite("loss_function", &Model::loss_function_)
//        .def_readwrite("regularizer", &Model::regularizer_)
//        .def_readwrite("reporter", &Model::reporter_)
//        .def_readwrite("current_device", &Model::current_device_)
//        .def_readwrite("devices", &Model::devices_)
//        .def_readwrite("learning_task", &Model::learning_task_)
//        .def_readwrite("model_config", &Model::model_config_)
//        .def_readwrite("has_embeddings", &Model::has_embeddings_)
//        .def_readwrite("has_features", &Model::has_features_)
//        .def_readwrite("reinitialize", &Model::reinitialize_)
//        .def_readwrite("is_train", &Model::train_)
//        .def_readwrite("filtered_eval", &Model::filtered_eval_)
//        .def_readwrite("device_models", &Model::device_models_)
//        .def(py::init<shared_ptr<ModelConfig>, shared_ptr<GeneralGNN>, shared_ptr<LossFunction>, shared_ptr<Regularizer>, shared_ptr<Featurizer>, shared_ptr<Reporter>>(),
//            py::arg("model_config"),
//            py::arg("encoder"),
//            py::arg("loss"),
//            py::arg("regularizer"),
//            py::arg("featurizer"),
//            py::arg("reporter") = nullptr)
//        .def("forward", &NodeClassificationModel::forward, py::arg("batch"), py::arg("train"))
//        .def("train_batch", static_cast<void (NodeClassificationModel::*)(Batch *)>(&NodeClassificationModel::train_batch), py::arg("batch"), py::call_guard<py::gil_scoped_release>())
//        .def("train_batch", static_cast<void (NodeClassificationModel::*)(std::vector<Batch *>)>(&NodeClassificationModel::train_batch), py::arg("sub_batches"), py::call_guard<py::gil_scoped_release>())
//        .def("evaluate", static_cast<void (NodeClassificationModel::*)(Batch *, bool)>(&NodeClassificationModel::evaluate), py::arg("batch"), py::arg("filtered_eval") = false, py::call_guard<py::gil_scoped_release>())
//        .def("evaluate", static_cast<void (NodeClassificationModel::*)(std::vector<Batch *>, bool)>(&NodeClassificationModel::evaluate), py::arg("sub_batches"), py::arg("filtered_eval") = false, py::call_guard<py::gil_scoped_release>());
//
//    torch::python::bind_module<LinkPredictionModel, Model>(m, "LinkPredictionModel")
//        .def_readwrite("featurizer", &Model::featurizer_)
//        .def_readwrite("featurizer_optimizer", &Model::featurizer_optimizer_)
//        .def_readwrite("encoder", &Model::encoder_)
//        .def_readwrite("encoder_optimizer", &Model::encoder_optimizer_)
//        .def_readwrite("decoder", &Model::decoder_)
//        .def_readwrite("decoder_optimizer", &Model::decoder_optimizer_)
//        .def_readwrite("loss_function", &Model::loss_function_)
//        .def_readwrite("regularizer", &Model::regularizer_)
//        .def_readwrite("reporter", &Model::reporter_)
//        .def_readwrite("current_device", &Model::current_device_)
//        .def_readwrite("devices", &Model::devices_)
//        .def_readwrite("learning_task", &Model::learning_task_)
//        .def_readwrite("model_config", &Model::model_config_)
//        .def_readwrite("has_embeddings", &Model::has_embeddings_)
//        .def_readwrite("has_features", &Model::has_features_)
//        .def_readwrite("reinitialize", &Model::reinitialize_)
//        .def_readwrite("is_train", &Model::train_)
//        .def_readwrite("filtered_eval", &Model::filtered_eval_)
//        .def_readwrite("device_models", &Model::device_models_)
//        .def(py::init<shared_ptr<ModelConfig>, shared_ptr<GeneralGNN>, shared_ptr<Decoder>, shared_ptr<LossFunction>, shared_ptr<Regularizer>, shared_ptr<Featurizer>, shared_ptr<Reporter>>(),
//            py::arg("model_config"),
//            py::arg("encoder"),
//            py::arg("decoder"),
//            py::arg("loss"),
//            py::arg("regularizer"),
//            py::arg("featurizer"),
//            py::arg("reporter") = nullptr)
//        .def("forward", &LinkPredictionModel::forward, py::arg("batch"), py::arg("train"))
//        .def("train_batch", static_cast<void (LinkPredictionModel::*)(Batch *)>(&LinkPredictionModel::train_batch), py::arg("batch"), py::call_guard<py::gil_scoped_release>())
//        .def("train_batch", static_cast<void (LinkPredictionModel::*)(std::vector<Batch *>)>(&LinkPredictionModel::train_batch), py::arg("sub_batches"), py::call_guard<py::gil_scoped_release>())
//        .def("evaluate", static_cast<void (LinkPredictionModel::*)(Batch *, bool)>(&LinkPredictionModel::evaluate), py::arg("batch"), py::arg("filtered_eval"), py::call_guard<py::gil_scoped_release>())
//        .def("evaluate", static_cast<void (LinkPredictionModel::*)(std::vector<Batch *>, bool)>(&LinkPredictionModel::evaluate), py::arg("sub_batches"), py::arg("filtered_eval"), py::call_guard<py::gil_scoped_release>());


//    m.def("initializeModel", &initializeModel, py::arg("model_config"), py::arg("devices"), py::arg("num_relations"));

    m.def("initializeModel", [](pyobj python_config, pybind11::list devices_pylist, int num_relations) {

        std::vector<torch::Device> devices = {};

        for (auto py_id: devices_pylist) {
            pyobj id_object = pybind11::reinterpret_borrow<pyobj>(py_id);
            devices.emplace_back(torch::python::detail::py_object_to_device(id_object));
        }

        return std::dynamic_pointer_cast<Model>(initializeModel(initModelConfig(python_config), devices, num_relations));

    }, py::arg("model_config"), py::arg("devices"), py::arg("num_relations"), py::return_value_policy::move);


    m.def("getOptimizerForModule", &getOptimizerForModule, py::arg("module"), py::arg("optimizer_config"));
}