//
// Created by Jason Mohoney on 2/15/22.
//

#include "common/pybind_headers.h"
#include "nn/decoders/edge/edge_decoder.h"

void init_edge_decoder(py::module &m) {
    py::class_<EdgeDecoder, Decoder, std::shared_ptr<EdgeDecoder>>(m, "EdgeDecoder")
        .def_readwrite("comparator", &EdgeDecoder::comparator_)
        .def_readwrite("relation_operator", &EdgeDecoder::relation_operator_)
        .def_readwrite("relations", &EdgeDecoder::relations_)
        .def_readwrite("inverse_relations", &EdgeDecoder::inverse_relations_)
        .def_readwrite("num_relations", &EdgeDecoder::num_relations_)
        .def_readwrite("embedding_size", &EdgeDecoder::embedding_size_)
        .def_readwrite("mode", &EdgeDecoder::decoder_method_)
        .def_readwrite("tensor_options", &EdgeDecoder::tensor_options_)
        .def_readwrite("use_inverse_relations", &EdgeDecoder::use_inverse_relations_)
        .def("apply_relation", &EdgeDecoder::apply_relation, py::arg("nodes"), py::arg("relations"))
        .def("compute_scores", &EdgeDecoder::apply_relation, py::arg("src"), py::arg("dst"))
        .def("select_relations", &EdgeDecoder::apply_relation, py::arg("indices"), py::arg("inverse") = false);
}