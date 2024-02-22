#include "common/pybind_headers.h"
#include "configuration/config.h"
#include "configuration/util.h"
#include "data/dataloader.h"
#include "storage/io.h"

void init_io(py::module &m) {
    m.def(
        "load_model",
        [](string filename, bool train) {
            shared_ptr<MariusConfig> marius_config = loadConfig(filename, train);

            std::vector<torch::Device> devices = devices_from_config(marius_config->storage);

            shared_ptr<Model> model = initModelFromConfig(marius_config->model, devices, marius_config->storage->dataset->num_relations, train);
            model->load(marius_config->storage->model_dir, train);

            return model;
        },
        py::arg("filename"), py::arg("train"));

    m.def(
        "load_storage",
        [](string filename, bool train) {
            shared_ptr<MariusConfig> marius_config = loadConfig(filename, train);

            std::vector<torch::Device> devices = devices_from_config(marius_config->storage);

            shared_ptr<Model> model = initModelFromConfig(marius_config->model, devices, marius_config->storage->dataset->num_relations, train);

            shared_ptr<GraphModelStorage> graph_model_storage = initializeStorage(model, marius_config->storage, false, train);

            return graph_model_storage;
        },
        py::arg("filename"), py::arg("train"));

    m.def(
        "init_from_config",
        [](string filename, bool train, bool load_storage) {
            shared_ptr<MariusConfig> marius_config = loadConfig(filename, train);

            std::vector<torch::Device> devices = devices_from_config(marius_config->storage);

            shared_ptr<Model> model = initModelFromConfig(marius_config->model, devices, marius_config->storage->dataset->num_relations, train);

            shared_ptr<GraphModelStorage> graph_model_storage = initializeStorage(model, marius_config->storage, true, train);

            shared_ptr<DataLoader> dataloader = std::make_shared<DataLoader>(graph_model_storage, model->learning_task_, marius_config->training,
                                                                             marius_config->evaluation, marius_config->model->encoder);

            if (train) {
                dataloader->setTrainSet();
            } else {
                dataloader->setTestSet();
            }

            dataloader->loadStorage();

            return std::make_tuple(model, dataloader);
        },
        py::arg("filename"), py::arg("train"), py::arg("load_storage") = true);
}
