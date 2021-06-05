#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "dataset.h"
#include "io.h"

namespace py = pybind11;

tuple<DataSet *, DataSet*> initializeDatasets(MariusOptions config) {

    DataSet *train_set;
    DataSet *eval_set;

    // set global configuration
    marius_options = config;

    tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> storage_ptrs = initializeTrain();

    Storage *train_edges;
    train_edges = std::get<0>(storage_ptrs);
    Storage *eval_edges = std::get<1>(storage_ptrs);
    Storage *test_edges = std::get<2>(storage_ptrs);

    Storage *embeddings = std::get<3>(storage_ptrs);
    Storage *emb_state = std::get<4>(storage_ptrs);

    Storage *lhs_rel = std::get<5>(storage_ptrs);
    Storage *lhs_rel_state = std::get<6>(storage_ptrs);
    Storage *rhs_rel = std::get<7>(storage_ptrs);
    Storage *rhs_rel_state = std::get<8>(storage_ptrs);

    bool will_train = !(marius_options.path.train_edges.empty());
    bool will_evaluate = !(marius_options.path.validation_edges.empty() && marius_options.path.test_edges.empty());

    if (will_train) {
        train_set = new DataSet(train_edges, embeddings, emb_state, lhs_rel, lhs_rel_state, rhs_rel, rhs_rel_state);
        SPDLOG_INFO("Training set initialized");
    }
    if (will_evaluate) {
        eval_set = new DataSet(train_edges, eval_edges, test_edges, embeddings, lhs_rel, rhs_rel);
        SPDLOG_INFO("Evaluation set initialized");
    }

    return forward_as_tuple(train_set, eval_set);
}
void init_io(py::module &m) {
    m.def("initializeDatasets", &initializeDatasets, py::return_value_policy::reference);
}
