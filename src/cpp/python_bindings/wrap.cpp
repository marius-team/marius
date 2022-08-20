#define PYBIND11_COMPILER_TYPE ""
#define PYBIND11_STDLIB ""
#define PYBIND11_BUILD_ABI ""

#include "torch/extension.h"

namespace py = pybind11;

// configuration
void init_config(py::module &);
void init_options(py::module &);

// decoders
//void init_comparators(py::module &);
//void init_complex(py::module &);
//void init_decoder(py::module &);
//void init_distmult(py::module &);
//void init_relation_operators(py::module &);
//void init_transe(py::module &);

// encoders
//void init_encoder(py::module &);
//void init_gat(py::module &);
//void init_gcn(py::module &);
//void init_gnn(py::module &);
//void init_graph_sage(py::module &);
//void init_rgcn(py::module &);

// featurizers
//void init_featurizer(py::module &);

// layers
//void init_gat_layer(py::module &);
//void init_gcn_layer(py::module &);
//void init_gnn_layer(py::module &);
//void init_graph_sage_layer(py::module &);
//void init_layer_helpers(py::module &);
//void init_rgcn_layer(py::module &);

void init_activation(py::module &);
void init_batch(py::module &);
void init_dataloader(py::module &);
void init_datatypes(py::module &);
void init_evaluator(py::module &);
void init_graph_samplers(py::module &);
void init_graph_storage(py::module &);
void init_graph(py::module &);
void init_initialization(py::module &);
void init_io(py::module &);
void init_loss(py::module &);
void init_marius(py::module &);
void init_model(py::module &);
void init_regularizer(py::module &);
void init_reporting(py::module &);
void init_trainer(py::module &);

PYBIND11_MODULE(_pymarius, m) {

	m.doc() = "pybind11 marius plugin";

    // configuration
    init_config(m);
    init_options(m);

    // decoders
//    init_comparators(m);
//    init_complex(m);
//    init_decoder(m);
//    init_distmult(m);
//    init_relation_operators(m);
//    init_transe(m);

    // encoders
//    init_gnn(m);

    // featurizers
//    init_featurizer(m);

    // layers
//    init_gat_layer(m);
//    init_gcn_layer(m);
//    init_gnn_layer(m);
//    init_graph_sage_layer(m);
//    init_layer_helpers(m);
//    init_rgcn_layer(m);

    init_activation(m);
    init_batch(m);
    init_dataloader(m);
    init_datatypes(m);
    init_evaluator(m);
    init_graph_samplers(m);
    init_graph_storage(m);
    init_graph(m);
    init_initialization(m);
    init_io(m);
    init_loss(m);
    init_marius(m);
    init_model(m);
    init_regularizer(m);
    init_reporting(m);
    init_trainer(m);
}