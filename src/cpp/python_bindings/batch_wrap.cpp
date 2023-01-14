#include <torch/extension.h>

#include "batch.h"

namespace py = pybind11;

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

	py::class_<Batch>(m, "Batch")
        .def_readwrite("batch_id", &Batch::batch_id_)
        .def_readwrite("start_idx", &Batch::start_idx_)
        .def_readwrite("batch_size", &Batch::batch_size_)
        .def_readwrite("train", &Batch::train_)
        .def_readwrite("device_id", &Batch::device_id_)
        
        .def_readwrite("load_timestamp", &Batch::load_timestamp_)
        .def_readwrite("compute_timestamp", &Batch::compute_timestamp_)
        //.def_readwrite("device_transfer", &Batch::device_transfer_)
        //.def_readwrite("host_transfer_", &Batch::host_transfer_)
        .def_readwrite("timer", &Batch::timer_)
        .def_readwrite("status", &Batch::status_)

        .def_readwrite("root_node_indices", &Batch::root_node_indices_)
        .def_readwrite("unique_node_indices", &Batch::unique_node_indices_)
        .def_readwrite("unique_node_embeddings", &Batch::unique_node_embeddings_)
        .def_readwrite("unique_node_gradients", &Batch::unique_node_gradients_)
        .def_readwrite("unique_node_embeddings_state", &Batch::unique_node_embeddings_state_)
        .def_readwrite("unique_node_state_update", &Batch::unique_node_state_update_)

        .def_readwrite("unique_node_features", &Batch::unique_node_features_)
        .def_readwrite("unique_node_labels", &Batch::unique_node_labels_)

        .def_readwrite("src_pos_indices_mapping", &Batch::src_pos_indices_mapping_)
        .def_readwrite("dst_pos_indices_mapping", &Batch::dst_pos_indices_mapping_)
        .def_readwrite("src_neg_indices_mapping", &Batch::src_neg_indices_mapping_)
        .def_readwrite("dst_neg_indices_mapping", &Batch::dst_neg_indices_mapping_)

        .def_readwrite("src_pos_indices", &Batch::src_pos_indices_)
        .def_readwrite("dst_pos_indices", &Batch::dst_pos_indices_)
        .def_readwrite("rel_indices", &Batch::rel_indices_)
        .def_readwrite("src_neg_indices", &Batch::src_neg_indices_)
        .def_readwrite("dst_neg_indices", &Batch::dst_neg_indices_)
        
        .def_readwrite("negative_sampling", &Batch::negative_sampling_)

        .def_readwrite("gnn_graph", &Batch::gnn_graph_)
        .def_readwrite("encoded_uniques", &Batch::encoded_uniques_)

        .def_readwrite("src_pos_embeddings", &Batch::src_pos_embeddings_)
        .def_readwrite("dst_pos_embeddings", &Batch::dst_pos_embeddings_)
        .def_readwrite("src_global_neg_embeddings", &Batch::src_global_neg_embeddings_)
        .def_readwrite("dst_global_neg_embeddings", &Batch::dst_global_neg_embeddings_)
        .def_readwrite("src_all_neg_embeddings", &Batch::src_all_neg_embeddings_)
        .def_readwrite("dst_all_neg_embeddings", &Batch::dst_all_neg_embeddings_)

        .def_readwrite("src_neg_filter", &Batch::src_neg_filter_)
        .def_readwrite("dst_neg_filter", &Batch::dst_neg_filter_)

        .def_readwrite("src_neg_filter_eval", &Batch::src_neg_filter_eval_)
        .def_readwrite("dst_neg_filter_eval", &Batch::dst_neg_filter_eval_)
        .def(py::init<bool>(), py::arg("train"))
        .def(py::init<std::vector<Batch *>>(), py::arg("sub_batches"))
        .def("setUniqueNodes", &Batch::setUniqueNodes, py::arg("use_neighbors") = false, py::arg("set_mapping") = false)
        .def("localSample", &Batch::localSample)
        .def("to", [](Batch &batch, torch::Device device) {
            batch.to(device, nullptr);
        }, py::arg("device"))
        .def("prepareBatch", &Batch::prepareBatch)
        .def("accumulateGradients", &Batch::accumulateGradients, py::arg("learning_rate"))
        .def("embeddingsToHost", &Batch::embeddingsToHost)
        .def("clear", &Batch::clear);
}