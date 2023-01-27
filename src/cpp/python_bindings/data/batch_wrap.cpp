#include "common/pybind_headers.h"
#include "data/batch.h"

void init_batch(py::module &m) {
    py::enum_<BatchStatus>(m, "BatchStatus")
        .value("Waiting", BatchStatus::Waiting)
        .value("AccumulatedIndices", BatchStatus::AccumulatedIndices)
        .value("LoadedEmbeddings", BatchStatus::LoadedEmbeddings)
        .value("TransferredToDevice", BatchStatus::TransferredToDevice)
        .value("PreparedForCompute", BatchStatus::PreparedForCompute)
        .value("ComputedGradients", BatchStatus::ComputedGradients)
        .value("AccumulatedGradients", BatchStatus::AccumulatedGradients)
        .value("TransferredToHost", BatchStatus::TransferredToHost)
        .value("Done", BatchStatus::Done);

    py::class_<Batch, shared_ptr<Batch>>(m, "Batch", py::dynamic_attr())
        .def_readwrite("batch_id", &Batch::batch_id_)
        .def_readwrite("start_idx", &Batch::start_idx_)
        .def_readwrite("batch_size", &Batch::batch_size_)
        .def_readwrite("train", &Batch::train_)
        .def_readwrite("device_id", &Batch::device_id_)

        .def_readwrite("status", &Batch::status_)

        .def_readwrite("root_node_indices", &Batch::root_node_indices_)
        .def_readwrite("unique_node_indices", &Batch::unique_node_indices_)
        .def_readwrite("node_embeddings", &Batch::node_embeddings_)
        .def_readwrite("node_gradients", &Batch::node_gradients_)
        .def_readwrite("node_embeddings_state", &Batch::node_embeddings_state_)
        .def_readwrite("node_state_update", &Batch::node_state_update_)

        .def_readwrite("node_features", &Batch::node_features_)
        .def_readwrite("node_labels", &Batch::node_labels_)

        .def_readwrite("src_neg_indices_mapping", &Batch::src_neg_indices_mapping_)
        .def_readwrite("dst_neg_indices_mapping", &Batch::dst_neg_indices_mapping_)

        .def_readwrite("edges", &Batch::edges_)

        .def_readwrite("dense_graph", &Batch::dense_graph_)
        .def_readwrite("encoded_uniques", &Batch::encoded_uniques_)

        .def_readwrite("neg_edges", &Batch::neg_edges_)
        .def_readwrite("rel_neg_indices", &Batch::rel_neg_indices_)
        .def_readwrite("src_neg_indices", &Batch::src_neg_indices_)
        .def_readwrite("dst_neg_indices", &Batch::dst_neg_indices_)
        .def_readwrite("src_neg_filter", &Batch::src_neg_filter_)
        .def_readwrite("dst_neg_filter", &Batch::dst_neg_filter_)

        .def(py::init<bool>(), py::arg("train"))
        .def("to", &Batch::to, py::arg("device"), py::arg("stream") = nullptr)
        .def("accumulateGradients", &Batch::accumulateGradients, py::arg("learning_rate"))
        .def("embeddingsToHost", &Batch::embeddingsToHost)
        .def("clear", &Batch::clear);
}