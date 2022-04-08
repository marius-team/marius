//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"

#include "nn/decoders/edge/complex.h"

void init_complex(py::module &m) {
    py::class_<ComplEx, EdgeDecoder, std::shared_ptr<ComplEx>>(m, "ComplEx")
        .def(py::init([](int num_relations, int embedding_dim, bool use_inverse_relations, py::object py_device, py::object py_dtype, string decoder_method) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return std::make_shared<ComplEx>(num_relations, embedding_dim, options, use_inverse_relations, getEdgeDecoderMethod(decoder_method));
        }), py::arg("num_relations"), py::arg("embedding_dim"), py::arg("use_inverse_relations") = true, py::arg("device") = py::none(), py::arg("dtype") = py::none(), py::arg("decoder_method") = "corrupt_node")
        .def("reset", &ComplEx::reset);
}