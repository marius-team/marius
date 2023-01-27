#include "common/pybind_headers.h"
#include "storage/graph_storage.h"

void init_graph_storage(py::module &m) {
    py::class_<GraphModelStoragePtrs, std::shared_ptr<GraphModelStoragePtrs>>(m, "GraphModelStoragePtrs")
        .def_readwrite("edges", &GraphModelStoragePtrs::edges)
        .def_readwrite("train_edges", &GraphModelStoragePtrs::train_edges)
        .def_readwrite("validation_edges", &GraphModelStoragePtrs::validation_edges)
        .def_readwrite("test_edges", &GraphModelStoragePtrs::test_edges)
        .def_readwrite("nodes", &GraphModelStoragePtrs::nodes)
        .def_readwrite("train_nodes", &GraphModelStoragePtrs::train_nodes)
        .def_readwrite("valid_nodes", &GraphModelStoragePtrs::valid_nodes)
        .def_readwrite("test_nodes", &GraphModelStoragePtrs::test_nodes)
        .def_readwrite("node_features", &GraphModelStoragePtrs::node_features)
        .def_readwrite("node_labels", &GraphModelStoragePtrs::node_labels)
        .def_readwrite("relation_features", &GraphModelStoragePtrs::relation_features)
        .def_readwrite("relation_labels", &GraphModelStoragePtrs::relation_labels)
        .def_readwrite("node_embeddings", &GraphModelStoragePtrs::node_embeddings)
        .def_readwrite("node_optimizer_state", &GraphModelStoragePtrs::node_optimizer_state);

    py::class_<InMemorySubgraphState, std::shared_ptr<InMemorySubgraphState>>(m, "InMemorySubgraphState")
        .def_readwrite("all_in_memory_edges", &InMemorySubgraphState::all_in_memory_edges_)
        .def_readwrite("all_in_memory_mapped_edges", &InMemorySubgraphState::all_in_memory_mapped_edges_)
        .def_readwrite("in_memory_partition_ids", &InMemorySubgraphState::in_memory_partition_ids_)
        .def_readwrite("in_memory_edge_bucket_ids", &InMemorySubgraphState::in_memory_edge_bucket_ids_)
        .def_readwrite("in_memory_edge_bucket_starts", &InMemorySubgraphState::in_memory_edge_bucket_starts_)
        .def_readwrite("in_memory_edge_bucket_sizes", &InMemorySubgraphState::in_memory_edge_bucket_sizes_)
        .def_readwrite("global_to_local_index_map", &InMemorySubgraphState::global_to_local_index_map_)
        .def_readwrite("in_memory_subgraph", &InMemorySubgraphState::in_memory_subgraph_);

    py::class_<GraphModelStorage, std::shared_ptr<GraphModelStorage>>(m, "GraphModelStorage")
        .def_readwrite("active_edges", &GraphModelStorage::active_edges_)
        .def_readwrite("active_nodes", &GraphModelStorage::active_nodes_)
        .def_readwrite("storage_ptrs", &GraphModelStorage::storage_ptrs_)
        .def_readwrite("full_graph_evaluation", &GraphModelStorage::full_graph_evaluation_)
        .def_readwrite("current_subgraph_state", &GraphModelStorage::current_subgraph_state_)
        .def_readwrite("next_subgraph_state", &GraphModelStorage::next_subgraph_state_)

        .def(py::init<GraphModelStoragePtrs, shared_ptr<StorageConfig>>(), py::arg("storage_ptrs"), py::arg("storage_config"))
        .def(py::init([](shared_ptr<Storage> edges, shared_ptr<Storage> nodes, shared_ptr<Storage> node_features, shared_ptr<Storage> node_embeddings,
                         shared_ptr<Storage> node_optimizer_state, shared_ptr<Storage> node_labels, std::vector<shared_ptr<Storage>> filter_edges, bool train,
                         bool prefetch) {
                 GraphModelStoragePtrs ptrs;
                 ptrs.edges = edges;
                 ptrs.nodes = nodes;
                 ptrs.node_features = node_features;
                 ptrs.node_embeddings = node_embeddings;
                 ptrs.node_optimizer_state = node_optimizer_state;
                 ptrs.node_labels = node_labels;
                 ptrs.filter_edges = filter_edges;

                 // initialize optimizer state if needed
                 if (train && node_optimizer_state == nullptr && node_embeddings != nullptr) {
                     string optimizer_state_filename = get_directory(node_embeddings->filename_);

                     shared_ptr<FlatFile> init_optimizer_state_storage = std::make_shared<FlatFile>(optimizer_state_filename, node_embeddings->dtype_);

                     int64_t curr_num_nodes = 0;
                     int64_t offset = 0;
                     int64_t num_nodes = node_embeddings->getDim0();

                     while (offset < num_nodes) {
                         if (num_nodes - offset < MAX_NODE_EMBEDDING_INIT_SIZE) {
                             curr_num_nodes = num_nodes - offset;
                         } else {
                             curr_num_nodes = MAX_NODE_EMBEDDING_INIT_SIZE;
                         }

                         OptimizerState emb_state = torch::zeros({curr_num_nodes, node_embeddings->dim1_size_}, node_embeddings->dtype_);
                         init_optimizer_state_storage->append(emb_state);

                         offset += curr_num_nodes;
                     }

                     if (instance_of<Storage, InMemory>(node_embeddings)) {
                         ptrs.node_optimizer_state = std::make_shared<InMemory>(optimizer_state_filename, node_embeddings->dtype_);
                     } else if (instance_of<Storage, PartitionBufferStorage>(node_embeddings)) {
                         ptrs.node_optimizer_state = std::make_shared<PartitionBufferStorage>(
                             optimizer_state_filename, std::dynamic_pointer_cast<PartitionBufferStorage>(node_embeddings)->options_);
                     } else {
                         throw MariusRuntimeException("Unsupported storage backend for embeddings");
                     }
                 }

                 return std::make_shared<GraphModelStorage>(ptrs, prefetch);
             }),
             py::arg("edges"), py::arg("nodes") = shared_ptr<Storage>(nullptr), py::arg("node_features") = shared_ptr<Storage>(nullptr),
             py::arg("node_embeddings") = shared_ptr<Storage>(nullptr), py::arg("node_optim_state") = shared_ptr<Storage>(nullptr),
             py::arg("node_labels") = shared_ptr<Storage>(nullptr), py::arg("filter_edges") = std::vector<shared_ptr<Storage>>(), py::arg("train") = false,
             py::arg("prefetch") = false)

        .def("load", &GraphModelStorage::load)
        .def("unload", &GraphModelStorage::unload, py::arg("write"))
        .def("init_subgraph", &GraphModelStorage::initializeInMemorySubGraph, py::arg("buffer_state"), py::arg("num_hash_maps") = 1)
        .def("update_subgraph", &GraphModelStorage::updateInMemorySubGraph)
        .def("sort_all_edges", &GraphModelStorage::sortAllEdges)
        .def("set_edge_storage", &GraphModelStorage::setEdgesStorage, py::arg("edge_storage"))
        .def("set_node_storage", &GraphModelStorage::setNodesStorage, py::arg("node_storage"))
        .def("get_edges", &GraphModelStorage::getEdges, py::arg("indices"))
        .def("get_edges_range", &GraphModelStorage::getEdgesRange, py::arg("start"), py::arg("size"))
        .def("getRandomNodeIds", &GraphModelStorage::getRandomNodeIds, py::arg("size"))
        .def("getNodeIdsRange", &GraphModelStorage::getNodeIdsRange, py::arg("start"), py::arg("size"))
        .def("shuffleEdges", &GraphModelStorage::shuffleEdges)
        .def("getNodeEmbeddings", &GraphModelStorage::getNodeEmbeddings, py::arg("indices"))
        .def("getNodeEmbeddingsRange", &GraphModelStorage::getNodeEmbeddingsRange, py::arg("start"), py::arg("size"))
        .def("getNodeFeatures", &GraphModelStorage::getNodeFeatures, py::arg("indices"))
        .def("getNodeFeaturesRange", &GraphModelStorage::getNodeFeaturesRange, py::arg("start"), py::arg("size"))
        .def("getNodeLabels", &GraphModelStorage::getNodeLabels, py::arg("indices"))
        .def("getNodeLabelsRange", &GraphModelStorage::getNodeLabelsRange, py::arg("start"), py::arg("size"))
        .def("updatePutNodeEmbeddings", &GraphModelStorage::updatePutNodeEmbeddings, py::arg("indices"), py::arg("embeddings"))
        .def("updateAddNodeEmbeddings", &GraphModelStorage::updateAddNodeEmbeddings, py::arg("indices"), py::arg("values"))
        .def("getNodeEmbeddingState", &GraphModelStorage::getNodeEmbeddingState, py::arg("indices"))
        .def("getNodeEmbeddingStateRange", &GraphModelStorage::getNodeEmbeddingStateRange, py::arg("start"), py::arg("size"))
        .def("updatePutNodeEmbeddingState", &GraphModelStorage::updatePutNodeEmbeddingState, py::arg("indices"), py::arg("state"))
        .def("updateAddNodeEmbeddingState", &GraphModelStorage::updateAddNodeEmbeddingState, py::arg("indices"), py::arg("values"))
        .def("embeddingsOffDevice", &GraphModelStorage::embeddingsOffDevice)
        .def("getNumPartitions", &GraphModelStorage::getNumPartitions)
        .def("useInMemorySubGraph", &GraphModelStorage::useInMemorySubGraph)
        .def("hasSwap", &GraphModelStorage::hasSwap)
        .def("performSwap", &GraphModelStorage::performSwap)
        .def("setBufferOrdering", &GraphModelStorage::setBufferOrdering, py::arg("buffer_states"))
        .def("setActiveEdges", &GraphModelStorage::setActiveEdges, py::arg("active_edges"))
        .def("setActiveNodes", &GraphModelStorage::setActiveNodes, py::arg("node_ids"))
        .def("getNumActiveEdges", &GraphModelStorage::getNumActiveEdges)
        .def("getNumActiveNodes", &GraphModelStorage::getNumActiveNodes)
        .def("getNumEdges", &GraphModelStorage::getNumEdges)
        .def("getNumNodes", &GraphModelStorage::getNumNodes)
        .def("getNumNodesInMemory", &GraphModelStorage::getNumNodesInMemory)
        .def("setTrainSet", &GraphModelStorage::setTrainSet)
        .def("setValidationSet", &GraphModelStorage::setValidationSet)
        .def("setTestSet", &GraphModelStorage::setTestSet)
        .def("setFilterEdges", &GraphModelStorage::setFilterEdges)
        .def("addFilterEdges", &GraphModelStorage::addFilterEdges);
}