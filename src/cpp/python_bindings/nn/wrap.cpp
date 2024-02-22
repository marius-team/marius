#include <pybind11/embed.h>

#include "common/pybind_headers.h"

namespace py = pybind11;

// nn
void init_activation(py::module &);
void init_initialization(py::module &);
void init_loss(py::module &);
void init_model(py::module &);
void init_optim(py::module &);
void init_regularizer(py::module &);

// nn/decoders
void init_decoder(py::module &);

// nn/decoders/edge
void init_comparators(py::module &);
void init_complex(py::module &);
void init_distmult(py::module &);
void init_edge_decoder(py::module &);
void init_relation_operators(py::module &);
void init_transe(py::module &);

// nn/decoders/node
void init_node_decoder(py::module &);
void init_noop_node_decoder(py::module &);

// nn/encoders
void init_encoder(py::module &);

// nn/layers
void init_layer(py::module &);

// nn/layers/dense

// nn/layers/embedding
void init_embedding_layer(py::module &);

// nn/layers/feature
void init_feature_layer(py::module &);

// nn/layers/gnn
void init_gat_layer(py::module &);
void init_gcn_layer(py::module &);
void init_gnn_layer(py::module &);
void init_graph_sage_layer(py::module &);
void init_layer_helpers(py::module &);
void init_rgcn_layer(py::module &);

// nn/layers/reduction
void init_concat_reduction_layer(py::module &);
void init_linear_reduction_layer(py::module &);
void init_reduction_layer(py::module &);

PYBIND11_MODULE(_nn, m) {
    m.doc() = "Contains model encoders, decoders and layers.";

    // nn
    init_activation(m);
    init_initialization(m);
    init_loss(m);
    init_model(m);
    init_optim(m);
    init_regularizer(m);

    // nn/decoders
    auto decoders_m = m.def_submodule("decoders");
    decoders_m.doc() = "Decoder models";

    init_decoder(decoders_m);

    // nn/decoders/edge
    auto edge_m = decoders_m.def_submodule("edge");
    edge_m.doc() = "Decoders for link prediction";

    init_edge_decoder(edge_m);
    init_comparators(edge_m);
    init_complex(edge_m);
    init_distmult(edge_m);
    init_relation_operators(edge_m);
    init_transe(edge_m);

    // nn/decoders/node
    auto node_m = decoders_m.def_submodule("node");
    node_m.doc() = "Decoders for node classification";

    init_node_decoder(node_m);
    init_noop_node_decoder(node_m);

    // nn/encoders
    auto encoders_m = m.def_submodule("encoders");
    encoders_m.doc() = "Model encoders";

    init_encoder(encoders_m);

    // nn/layers
    auto layers_m = m.def_submodule("layers");
    layers_m.doc() = "Layers for encoders";
    init_layer(layers_m);

    // nn/layers/dense

    // nn/layers/embedding
    init_embedding_layer(layers_m);

    // nn/layers/feature
    init_feature_layer(layers_m);

    // nn/layers/gnn
    init_gnn_layer(layers_m);
    init_gat_layer(layers_m);
    init_gcn_layer(layers_m);
    init_graph_sage_layer(layers_m);
    init_layer_helpers(layers_m);
    init_rgcn_layer(layers_m);

    // nn/layers/reduction
    init_reduction_layer(layers_m);
    init_concat_reduction_layer(layers_m);
    init_linear_reduction_layer(layers_m);
}
