#include "common/pybind_headers.h"

// configuration
void init_config(py::module &);
void init_options(py::module &);

PYBIND11_MODULE(_config, m) {
    m.doc() = "Configuration and options for API objects.";

    // configuration
    init_config(m);
    init_options(m);
}
