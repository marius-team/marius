#include "common/pybind_headers.h"

// pipeline
void init_evaluator(py::module &);
void init_graph_encoder(py::module &);
void init_trainer(py::module &);

PYBIND11_MODULE(_pipeline, m) {
    m.doc() = "Training and Evaluation pipelines.";

    // pipeline
    init_evaluator(m);
    init_graph_encoder(m);
    init_trainer(m);
}