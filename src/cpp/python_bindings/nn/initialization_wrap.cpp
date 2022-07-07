#include "common/pybind_headers.h"
#include "nn/initialization.h"

void init_initialization(py::module &m) {
    m.def("compute_fans", &compute_fans, py::arg("shape"));

    m.def(
        "glorot_uniform",
        [](std::vector<int64_t> shape, py::object py_device, py::object py_dtype, std::tuple<int64_t, int64_t> fans) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return glorot_uniform(shape, fans, options);
        },
        py::arg("shape"), py::arg("device"), py::arg("dtype"), py::arg("fans") = std::make_tuple(-1, -1));

    m.def(
        "glorot_normal",
        [](std::vector<int64_t> shape, py::object py_device, py::object py_dtype, std::tuple<int64_t, int64_t> fans) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return glorot_normal(shape, fans, options);
        },
        py::arg("shape"), py::arg("device"), py::arg("dtype"), py::arg("fans") = std::make_tuple(-1, -1));

    m.def(
        "constant_init",
        [](std::vector<int64_t> shape, float constant, py::object py_device, py::object py_dtype) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return constant_init(constant, shape, options);
        },
        py::arg("shape"), py::arg("constant") = 0, py::arg("device"), py::arg("dtype"));

    m.def(
        "uniform_init",
        [](std::vector<int64_t> shape, float scale_factor, py::object py_device, py::object py_dtype) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return uniform_init(scale_factor, shape, options);
        },
        py::arg("shape"), py::arg("scale_factor") = .001, py::arg("device"), py::arg("dtype"));

    m.def(
        "normal_init",
        [](std::vector<int64_t> shape, float mean, float std, py::object py_device, py::object py_dtype) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return normal_init(mean, std, shape, options);
        },
        py::arg("shape"), py::arg("mean") = 0, py::arg("std") = 1, py::arg("device"), py::arg("dtype"));

    m.def(
        "initialize_tensor",
        [](shared_ptr<InitConfig> init_config, std::vector<int64_t> shape, py::object py_device, py::object py_dtype, std::tuple<int64_t, int64_t> fans) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return initialize_tensor(init_config, shape, options, fans);
        },
        py::arg("init_config"), py::arg("shape"), py::arg("device"), py::arg("dtype"), py::arg("fans") = std::make_tuple(-1, -1));

    m.def(
        "initialize_subtensor",
        [](shared_ptr<InitConfig> init_config, std::vector<int64_t> sub_shape, std::vector<int64_t> full_shape, py::object py_device, py::object py_dtype,
           std::tuple<int64_t, int64_t> fans) {
            torch::TensorOptions options;
            options = options.device(torch::python::detail::py_object_to_device(py_device)).dtype(torch::python::detail::py_object_to_dtype(py_dtype));
            return initialize_subtensor(init_config, sub_shape, full_shape, options, fans);
        },
        py::arg("init_config"), py::arg("sub_shape"), py::arg("full_shape"), py::arg("device"), py::arg("dtype"), py::arg("fans") = std::make_tuple(-1, -1));
}