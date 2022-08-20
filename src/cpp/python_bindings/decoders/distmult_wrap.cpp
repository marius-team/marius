
#include "decoders/distmult.h"

#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/python.h>

void init_distmult(py::module &m) {
    
    torch::python::bind_module<DistMult>(m, "DistMult")
            .def(py::init([](int num_relations, int embedding_dim, bool use_inverse_relations, py::object py_device, py::object py_dtype) {
                auto options = torch::TensorOptions().device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
                return std::unique_ptr<DistMult>(new DistMult(num_relations, embedding_dim, options, use_inverse_relations));
            }), py::arg("num_relations"), py::arg("embedding_dim"), py::arg("use_inverse_relations") = true, py::arg("device") = nullptr, py::arg("dtype") = nullptr)
            .def("reset", &DistMult::reset);
}

