#include "common/pybind_headers.h"

// reporting
void init_reporting(py::module &);

PYBIND11_MODULE(_report, m) {
    m.doc() = "Training and evaluation metrics.";

    // reporting
    init_reporting(m);
}
