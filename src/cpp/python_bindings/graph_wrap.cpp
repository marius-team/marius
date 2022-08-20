#include <torch/extension.h>

#include "graph.h"

namespace py = pybind11;

void init_graph(py::module &m) {

    py::class_<MariusGraph>(m, "MariusGraph")
        .def_readwrite("src_sorted_edges", &MariusGraph::src_sorted_edges_)
        .def_readwrite("dst_sorted_edges", &MariusGraph::dst_sorted_edges_)
        .def_readwrite("active_in_memory_subgraph", &MariusGraph::active_in_memory_subgraph_)
        .def_readwrite("num_nodes_in_memory", &MariusGraph::num_nodes_in_memory_)
        .def_readwrite("node_ids", &MariusGraph::node_ids_)
        .def_readwrite("out_sorted_uniques", &MariusGraph::out_sorted_uniques_)
        .def_readwrite("out_offsets", &MariusGraph::out_offsets_)
        .def_readwrite("out_num_neighbors", &MariusGraph::out_num_neighbors_)
        .def_readwrite("in_sorted_uniques", &MariusGraph::in_sorted_uniques_)
        .def_readwrite("in_offsets", &MariusGraph::in_offsets_)
        .def_readwrite("in_num_neighbors", &MariusGraph::in_num_neighbors_)
        .def_readwrite("max_out_num_neighbors_", &MariusGraph::max_out_num_neighbors_)
        .def_readwrite("max_in_num_neighbors_", &MariusGraph::max_in_num_neighbors_)
        .def(py::init<>())
        .def(py::init<EdgeList, EdgeList, int64_t>(), py::arg("src_sorted_edges"), py::arg("dst_sorted_edges"), py::arg("num_nodes_in_memory"))
        .def("getEdges", &MariusGraph::getEdges, py::arg("incoming") = true)
        .def("getRelationIDs", &MariusGraph::getRelationIDs, py::arg("incoming") = true)
        .def("getNeighborOffsets", &MariusGraph::getNeighborOffsets, py::arg("incoming") = true)
        .def("getNumNeighbors", &MariusGraph::getNumNeighbors, py::arg("incoming") = true)
        .def("getNeighborsForNodeIds", &MariusGraph::getNeighborsForNodeIds, py::arg("node_ids"), py::arg("incoming"), py::arg("neighbor_sampling_layer"), py::arg("max_neighbors_size"), py::arg("rate"))
        .def("clear", &MariusGraph::clear)
        .def("to", &MariusGraph::to, py::arg("device"));

    py::class_<GNNGraph, MariusGraph>(m, "GNNGraph")
        .def_readwrite("hop_offsets", &GNNGraph::hop_offsets_)
        .def_readwrite("in_neighbors_mapping", &GNNGraph::in_neighbors_mapping_)
        .def_readwrite("out_neighbors_mapping", &GNNGraph::out_neighbors_mapping_)
        .def_readwrite("in_neighbors_vec", &GNNGraph::in_neighbors_vec_)
        .def_readwrite("out_neighbors_vec", &GNNGraph::out_neighbors_vec_)
        .def_readwrite("node_properties", &GNNGraph::node_properties_)
        .def_readwrite("num_nodes_in_memory", &GNNGraph::num_nodes_in_memory_)
        .def(py::init<>())
        .def(py::init<Indices, Indices, Indices, std::vector<torch::Tensor>, Indices, Indices, std::vector<torch::Tensor>, Indices, int>(), 
            py::arg("hop_offsets"), 
            py::arg("node_ids"), 
            py::arg("in_offsets"), 
            py::arg("in_neighbors_vec"), 
            py::arg("in_neighbors_mapping"), 
            py::arg("out_offsets"), 
            py::arg("out_neighbors_vec"), 
            py::arg("out_neighbors_mapping"), 
            py::arg("num_nodes_in_memory"))
        .def("prepareForNextLayer", &GNNGraph::prepareForNextLayer)
        .def("getNeighborIDs", &GNNGraph::getNeighborIDs, py::arg("incoming") = true, py::arg("global") = false)
        .def("getLayerOffset", &GNNGraph::getLayerOffset)
        .def("performMap", &GNNGraph::performMap)
        .def("setNodeProperties", &GNNGraph::setNodeProperties, py::arg("node_properties"))
        .def("clear", &GNNGraph::clear)
        .def("to", &GNNGraph::to, py::arg("device"));
}