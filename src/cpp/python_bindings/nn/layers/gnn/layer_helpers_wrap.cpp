
#include "common/pybind_headers.h"
#include "nn/layers/gnn/layer_helpers.h"

void init_layer_helpers(py::module &m) {
    m.def("segment_ids_from_offsets", &segment_ids_from_offsets, py::arg("offsets"), py::arg("input_size"));

    m.def("segmented_sum", &segmented_sum, py::arg("tensor"), py::arg("segment_ids"), py::arg("num_segments"));

    m.def("segmented_sum_with_offsets", &segmented_sum_with_offsets, py::arg("tensor"), py::arg("offsets"));

    m.def("segmented_max_with_offsets", &segmented_max_with_offsets, py::arg("tensor"), py::arg("offsets"));

    m.def("attention_softmax", &attention_softmax, py::arg("neighbor_attention"), py::arg("self_attention"), py::arg("segment_offsets"), py::arg("segment_ids"),
          py::arg("num_nbrs"));
}