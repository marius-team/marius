//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/decoders/edge/transe.h"

void init_transe(py::module &m) {
    py::class_<TransE, EdgeDecoder, std::shared_ptr<TransE>>(m, "TransE")
        .def(py::init([](int num_relations, int embedding_dim, bool use_inverse_relations, py::object py_device, py::object py_dtype, string decoder_method) {
                 torch::TensorOptions options;
                 options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
                 return std::make_shared<TransE>(num_relations, embedding_dim, options, use_inverse_relations, getEdgeDecoderMethod(decoder_method));
             }),
             py::arg("num_relations"), py::arg("embedding_dim"), py::arg("use_inverse_relations") = true, py::arg("device") = py::none(),
             py::arg("dtype") = py::none(), py::arg("mode") = "train")
        .def("reset", &TransE::reset);
}
