#include <sys/stat.h>

#include "common/pybind_headers.h"
#include "storage/storage.h"

// Trampoline class
class PyStorage : Storage {
   public:
    using Storage::Storage;

    torch::Tensor indexRead(torch::Tensor indices) override { PYBIND11_OVERRIDE_PURE(torch::Tensor, Storage, indexRead, indices); }

    void indexAdd(torch::Tensor indices, torch::Tensor values) override { PYBIND11_OVERRIDE_PURE(void, Storage, indexAdd, indices, values); }

    torch::Tensor range(int64_t offset, int64_t n) override { PYBIND11_OVERRIDE_PURE(torch::Tensor, Storage, range, offset, n); }

    void indexPut(torch::Tensor indices, torch::Tensor values) override { PYBIND11_OVERRIDE_PURE(void, Storage, indexPut, indices, values); }

    void rangePut(int64_t offset, int64_t n, torch::Tensor values) override { PYBIND11_OVERRIDE_PURE(void, Storage, rangePut, offset, n, values); }

    void load() override { PYBIND11_OVERRIDE_PURE(void, Storage, load); }

    void write() override { PYBIND11_OVERRIDE_PURE(void, Storage, write); }

    void unload(bool write) override { PYBIND11_OVERRIDE_PURE(void, Storage, unload, write); }

    void shuffle() override { PYBIND11_OVERRIDE_PURE(void, Storage, shuffle); }

    void sort(bool src) override { PYBIND11_OVERRIDE_PURE(void, Storage, sort, src); }
};

void init_storage(py::module &m) {
    py::class_<Storage, PyStorage, std::shared_ptr<Storage>>(m, "Storage")
        .def_readwrite("dim0_size", &Storage::dim0_size_)
        .def_readwrite("dim1_size", &Storage::dim1_size_)
        .def_readwrite("dtype", &Storage::dtype_)
        .def_readwrite("initialized", &Storage::initialized_)
        .def_readwrite("edge_bucket_sizes", &Storage::edge_bucket_sizes_)
        .def_readwrite("data", &Storage::data_)
        .def_readwrite("device", &Storage::device_)
        .def_readwrite("filename", &Storage::filename_)
        .def("indexRead", &Storage::indexRead, py::arg("indices"))
        .def("indexAdd", &Storage::indexAdd, py::arg("indices"), py::arg("values"))
        .def("range", &Storage::range, py::arg("offset"), py::arg("n"))
        .def("indexPut", &Storage::indexPut, py::arg("indices"), py::arg("values"))
        .def("rangePut", &Storage::rangePut, py::arg("offset"), py::arg("n"), py::arg("values"))
        .def("load", &Storage::load)
        .def("write", &Storage::write)
        .def("unload", &Storage::unload, py::arg("write"))
        .def("shuffle", &Storage::shuffle)
        .def("sort", &Storage::sort, py::arg("src"))
        .def("read_edge_bucket_sizes", &Storage::readPartitionSizes, py::arg("filename"));

    py::class_<PartitionBufferStorage, Storage, std::shared_ptr<PartitionBufferStorage>>(m, "PartitionBufferStorage")
        .def_readwrite("filename", &PartitionBufferStorage::filename_)
        .def_readwrite("loaded", &PartitionBufferStorage::loaded_)
        .def_readwrite("options", &PartitionBufferStorage::options_)
        .def(py::init<string, int64_t, int64_t, shared_ptr<PartitionBufferOptions>>(), py::arg("filename"), py::arg("dim0_size"), py::arg("dim1_size"),
             py::arg("options"))
        .def(py::init<string, torch::Tensor, shared_ptr<PartitionBufferOptions>>(), py::arg("filename"), py::arg("data"), py::arg("options"))
        .def(py::init<string, shared_ptr<PartitionBufferOptions>>(), py::arg("filename"), py::arg("options"))
        .def("hasSwap", &PartitionBufferStorage::hasSwap)
        .def("performNextSwap", &PartitionBufferStorage::performNextSwap)
        .def("getGlobalToLocalMap", &PartitionBufferStorage::getGlobalToLocalMap, py::arg("get_current") = true)
        .def("sync", &PartitionBufferStorage::sync)
        .def("setBufferOrdering", &PartitionBufferStorage::setBufferOrdering, py::arg("buffer_states"))
        .def("getNextAdmit", &PartitionBufferStorage::getNextAdmit)
        .def("getNextEvict", &PartitionBufferStorage::getNextEvict)
        .def("getNumInMemory", &PartitionBufferStorage::getNumInMemory);

    py::class_<FlatFile, Storage, std::shared_ptr<FlatFile>>(m, "FlatFile")
        .def(py::init([](std::string filename, std::vector<int64_t> shape, py::object py_dtype, bool alloc) {
                 int64_t dim0_size;
                 int64_t dim1_size;

                 if (shape.size() > 2 || shape.empty()) {
                     throw MariusRuntimeException("Tensor shape must be 1 or 2 dimensional.");
                 } else if (shape.size() == 2) {
                     dim0_size = shape[0];
                     dim1_size = shape[1];
                 } else {
                     dim0_size = shape[0];
                     dim1_size = 1;
                 }

                 torch::Dtype dtype = torch::python::detail::py_object_to_dtype(py_dtype);

                 return std::make_shared<FlatFile>(filename, dim0_size, dim1_size, dtype, alloc);
             }),
             py::arg("filename"), py::arg("shape"), py::arg("dtype"), py::arg("alloc") = false)

        .def(py::init<string, torch::Tensor>(), py::arg("filename"), py::arg("data"))

        .def(py::init([](std::string filename, py::object py_dtype) {
                 torch::Dtype dtype = torch::python::detail::py_object_to_dtype(py_dtype);

                 return std::make_shared<FlatFile>(filename, dtype);
             }),
             py::arg("filename"), py::arg("dtype"))

        .def("append", &FlatFile::append, py::arg("values"))
        .def("move", &FlatFile::move, py::arg("new_filename"))
        .def("copy", &FlatFile::copy, py::arg("new_filename"), py::arg("rename"))
        .def("mem_load", &FlatFile::mem_load)
        .def("mem_unload", &FlatFile::mem_unload, py::arg("write"));

    py::class_<InMemory, Storage, std::shared_ptr<InMemory>>(m, "InMemory")
        .def(py::init([](std::string filename, std::vector<int64_t> shape, py::object py_dtype, torch::Device device) {
                 int64_t dim0_size;
                 int64_t dim1_size;

                 if (shape.size() > 2 || shape.empty()) {
                     throw MariusRuntimeException("Tensor shape must be 1 or 2 dimensional.");
                 } else if (shape.size() == 2) {
                     dim0_size = shape[0];
                     dim1_size = shape[1];
                 } else {
                     dim0_size = shape[0];
                     dim1_size = 1;
                 }

                 torch::Dtype dtype = torch::python::detail::py_object_to_dtype(py_dtype);

                 return std::make_shared<InMemory>(filename, dim0_size, dim1_size, dtype, device);
             }),
             py::arg("filename"), py::arg("shape"), py::arg("dtype"), py::arg("device"))

        .def(py::init<string, torch::Tensor, torch::Device>(), py::arg("filename"), py::arg("data"), py::arg("device"))

        .def(py::init([](std::string filename, py::object py_dtype) {
                 torch::Dtype dtype = torch::python::detail::py_object_to_dtype(py_dtype);

                 return std::make_shared<InMemory>(filename, dtype);
             }),
             py::arg("filename"), py::arg("dtype"))

        .def(py::init<torch::Tensor>(), py::arg("data"));

    m.def(
        "tensor_from_file",
        [](py::object py_filename, std::vector<int64_t> shape, py::object py_dtype, py::object py_device) {
            std::string filename = py::str(((py::object)py_filename.attr("__str__"))());

            torch::Dtype dtype = torch::python::detail::py_object_to_dtype(py_dtype);
            torch::Device device = torch::python::detail::py_object_to_device(py_device);
            int dtype_size = get_dtype_size_wrapper(dtype);

            struct stat stat_buf;
            int rc = stat(filename.c_str(), &stat_buf);
            int64_t file_size = rc == 0 ? stat_buf.st_size : -1;

            if (file_size == -1) {
                throw MariusRuntimeException("Cannot get size of file: " + filename);
            }

            int64_t dim0_size = file_size / dtype_size;
            int64_t dim1_size = 1;

            auto storage = std::make_shared<InMemory>(filename, dim0_size, dim1_size, dtype, device);
            storage->load();
            return storage->data_.clone().reshape(shape);
        },
        py::arg("filename"), py::arg("shape"), py::arg("dtype"), py::arg("device"));
}