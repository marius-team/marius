#include "common/pybind_headers.h"

// storage
void init_graph_storage(py::module &);
void init_io(py::module &);
void init_storage(py::module &);

PYBIND11_MODULE(_storage, m) {
    m.doc() = "Storage objects for arbitrary backends.";

    // storage
    init_storage(m);
    init_graph_storage(m);
    init_io(m);
}
