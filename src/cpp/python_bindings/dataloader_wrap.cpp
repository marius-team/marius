#include <torch/extension.h>

#include "dataloader.h"

namespace py = pybind11;

void init_dataloader(py::module &m) {

    py::class_<DataLoader>(m, "DataLoader")
        .def_readwrite("graph_storage", &DataLoader::graph_storage_)
        .def_readwrite("edge_sampler", &DataLoader::edge_sampler_)
        .def_readwrite("negative_sampler", &DataLoader::negative_sampler_)
        .def_readwrite("neighbor_sampler", &DataLoader::neighbor_sampler_)
        .def_readwrite("training_config", &DataLoader::training_config_)
        .def_readwrite("evaluation_config", &DataLoader::evaluation_config_)
        .def(py::init<GraphModelStorage*, shared_ptr<TrainingConfig>, shared_ptr<EvaluationConfig>, shared_ptr<EncoderConfig>>(),
            py::arg("graph_storage"),
            py::arg("training_config"),
            py::arg("evaluation_config"),
            py::arg("encoder_config"))
        .def("setBufferOrdering", &DataLoader::setBufferOrdering)
        .def("setActiveEdges", &DataLoader::setActiveEdges)
        .def("setActiveNodes", &DataLoader::setActiveNodes)
        .def("initializeBatches", &DataLoader::initializeBatches)
        .def("clearBatches", &DataLoader::clearBatches)
        .def("hasNextBatch", &DataLoader::hasNextBatch)
        .def("getNextBatch", &DataLoader::getNextBatch)
        .def("finishedBatch", &DataLoader::finishedBatch)
        .def("getBatch", &DataLoader::getBatch, py::arg("worker_id"))
        .def("getSubBatches", &DataLoader::getSubBatches)
        .def("linkPredictionSample", &DataLoader::linkPredictionSample, py::arg("batch"), py::arg("worker_id"))
        .def("nodeClassificationSample", &DataLoader::nodeClassificationSample, py::arg("batch"), py::arg("worker_id"))
        .def("loadCPUParameters", &DataLoader::loadCPUParameters, py::arg("batch"))
        .def("loadGPUParameters", &DataLoader::loadGPUParameters, py::arg("batch"))
        .def("updateEmbeddingsForBatch", &DataLoader::updateEmbeddingsForBatch, py::arg("batch"), py::arg("gpu"))
        .def("nextEpoch", &DataLoader::nextEpoch)
        .def("loadStorage", &DataLoader::loadStorage)
        .def("epochComplete", &DataLoader::epochComplete)
        .def("unloadStorage", &DataLoader::unloadStorage, py::arg("write") = false)
        .def("getNumEdges", &DataLoader::getNumEdges)
        .def("getEpochsProcessed", &DataLoader::getEpochsProcessed)
        .def("getBatchesProcessed", &DataLoader::getBatchesProcessed)
        .def("isTrain", &DataLoader::isTrain)
        .def("setTrainSet", &DataLoader::setTrainSet)
        .def("setValidationSet", &DataLoader::setValidationSet)
        .def("setTestSet", &DataLoader::setTestSet);
}