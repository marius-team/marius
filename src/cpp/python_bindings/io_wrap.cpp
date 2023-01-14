#include <torch/extension.h>

 #include "configuration/config.h"
 #include "dataloader.h"
 #include "io.h"

 namespace py = pybind11;

 void init_io(py::module &m) {

     m.def("initializeEdges", [](pyobj python_config) {
         shared_ptr<StorageConfig> storage_config = initStorageConfig(python_config);
         return initializeTrainEdgesStorage(storage_config);
     }, py::arg("storage_config"));

     m.def("initialize_from_file", [](string filename, bool train, bool load_storage) {

         shared_ptr<MariusConfig> marius_config = initConfig(filename);

         std::vector<torch::Device> devices = {};

         if (marius_config->storage->device_type == torch::kCUDA) {
             for (int i = 0; i < marius_config->storage->device_ids.size(); i++) {
                 devices.emplace_back(torch::Device(torch::kCUDA, marius_config->storage->device_ids[i]));
             }

             if (devices.empty()) {
                 devices.emplace_back(torch::Device(torch::kCUDA, 0));
             }
         } else {
             devices.emplace_back(torch::kCPU);
         }

         std::shared_ptr<Model> model = initializeModel(marius_config->model,
                                                        devices,
                                                        marius_config->storage->dataset->num_relations);

         model->train_ = train;

         if (marius_config->evaluation->negative_sampling != nullptr) {
             model->filtered_eval_ = marius_config->evaluation->negative_sampling->filtered;
         }

         GraphModelStorage *graph_model_storage = initializeStorage(model, marius_config->storage);

         DataLoader *dataloader = new DataLoader(graph_model_storage,
                                                 marius_config->training,
                                                 marius_config->evaluation,
                                                 marius_config->model->encoder);

         if (train) {
             dataloader->setTrainSet();
         } else {
             dataloader->setTestSet();
         }

         if (load_storage) {
             dataloader->loadStorage();
         }

         return std::make_tuple(model, dataloader);
     }, py::arg("filename"),py::arg("train"), py::arg("load_storage") = true);

     m.def("initializeEdges", [](pyobj python_config) {
         shared_ptr<StorageConfig> storage_config = initStorageConfig(python_config);
         return initializeTrainEdgesStorage(storage_config);
     }, py::arg("storage_config"));

 }
