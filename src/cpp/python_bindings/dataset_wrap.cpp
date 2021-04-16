#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dataset.h>

namespace py = pybind11;

void init_dataset(py::module &m) {
    // bindings to DataSet class
    py::class_<DataSet>(m, "DataSet")
        .def(py::init<Storage*, Storage*, Storage*, Storage*, Storage*, Storage*, Storage*>(),
            py::arg("edges"), py::arg("embeddings"), py::arg("emb_state"), py::arg("lhs_relations"), py::arg("lhs_rel_state"),
            py::arg("rhs_relations"), py::arg("rhs_rel_state"))
        .def(py::init<Storage*, Storage*, Storage*, Storage*, Storage*, Storage*>(),
            py::arg("train_edges"), py::arg("eval_edges"), py::arg("test_edges"), py::arg("embeddings"), py::arg("lhs_relations"), py::arg("rhs_relations"))
        .def(py::init<Storage*, Storage*, Storage*, Storage*>(),
            py::arg("test_edges"), py::arg("embeddings"), py::arg("lhs_relations"), py::arg("rhs_relations"))
        .def("nextBatch", &DataSet::nextBatch)
        .def("initializeBatches", &DataSet::initializeBatches)
        .def("splitBatches", &DataSet::splitBatches)
        .def("clearBatches", &DataSet::clearBatches)
        .def("setEvalFilter", &DataSet::setEvalFilter, py::arg("batch"))
        .def("uniformIndices", py::overload_cast<>(&DataSet::uniformIndices))
        .def("uniformIndices", py::overload_cast<int>(&DataSet::uniformIndices))
        .def("shuffleIndices", &DataSet::shuffleIndices)
        .def("getNegativesIndices", py::overload_cast<>(&DataSet::getNegativesIndices))
        .def("getNegativesIndices", py::overload_cast<int>(&DataSet::getNegativesIndices))
        .def("getBatch", &DataSet::getBatch)
        .def("globalSample", &DataSet::globalSample, py::arg("batch"))
        .def("loadCPUParameters", &DataSet::loadCPUParameters, py::arg("batch"))
        .def("loadGPUParameters", &DataSet::loadGPUParameters, py::arg("batch"))
        .def("updateEmbeddingsForBatch", &DataSet::updateEmbeddingsForBatch, py::arg("batch"), py::arg("gpu"))
        .def("nextEpoch", &DataSet::nextEpoch)
        .def("checkpointParameters", &DataSet::checkpointParameters)
        .def("loadStorage", &DataSet::loadStorage)
        .def("unloadStorage", &DataSet::unloadStorage)
        .def("hasNextBatch", &DataSet::hasNextBatch)
        .def("isDone", &DataSet::isDone)
        .def("setTestSet", &DataSet::setTestSet)
        .def("setValidationSet", &DataSet::setValidationSet)
        .def("getDevice", &DataSet::getDevice) // need to cast back to pytorch device
        .def("getEpochsProcessed", &DataSet::getEpochsProcessed)
        .def("getSize", &DataSet::getNumEdges)
        .def("getNumBatches", &DataSet::getNumBatches)
        .def("getNumNodes", &DataSet::getNumNodes)
        .def("getBatchesProcessed", &DataSet::getBatchesProcessed)
        .def("getProgress", &DataSet::getProgress)
        .def("getTimestamp", &DataSet::getTimestamp) // implement timestamp
        .def("updateTimestamp", &DataSet::updateTimestamp)
        .def("isTrain", &DataSet::isTrain)
        .def("setCurrPos", &DataSet::setCurrPos)
        .def("syncEmbeddings", &DataSet::syncEmbeddings)
        .def("accumulateRanks", &DataSet::accumulateRanks) // convert to pytorch tensor
        .def("accumulateAuc", &DataSet::accumulateAuc);
}
