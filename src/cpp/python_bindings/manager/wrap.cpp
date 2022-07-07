#include "common/pybind_headers.h"

void init_marius(py::module &);

PYBIND11_MODULE(_manager, m) {
    m.doc() = "High level execution management.";

    // manager
    init_marius(m);
}
