#include "common/pybind_headers.h"
#include "data/dataloader.h"

void init_dataloader(py::module &m) {
    py::class_<DataLoader, shared_ptr<DataLoader>>(m, "DataLoader", py::dynamic_attr())
        .def_readwrite("graph_storage", &DataLoader::graph_storage_)
        .def_readwrite("edge_sampler", &DataLoader::edge_sampler_)
        .def_readwrite("negative_sampler", &DataLoader::negative_sampler_)
        .def_readwrite("neighbor_sampler", &DataLoader::neighbor_sampler_)
        .def_readwrite("training_config", &DataLoader::training_config_)
        .def_readwrite("evaluation_config", &DataLoader::evaluation_config_)
        .def_readwrite("train", &DataLoader::train_)
        .def_readwrite("epochs_processed", &DataLoader::epochs_processed_)
        .def_readwrite("batches_processed", &DataLoader::batches_processed_)
        .def_readwrite("current_edge", &DataLoader::current_edge_)
        .def_readwrite("batches", &DataLoader::batches_)
        .def_readwrite("batch_id_offset", &DataLoader::batch_id_offset_)
        //        .def_readwrite("batch_iterator", &DataLoader::batch_iterator_) # TODO Iterator needs bindings
        .def_readwrite("batches_left", &DataLoader::batches_left_)
        .def_readwrite("batches_processed", &DataLoader::total_batches_processed_)
        .def_readwrite("all_read", &DataLoader::all_read_)
        .def_readwrite("edge_buckets_per_buffer", &DataLoader::edge_buckets_per_buffer_)
        .def_readwrite("node_ids_per_buffer", &DataLoader::node_ids_per_buffer_)
        .def_readwrite("training_neighbor_sampler", &DataLoader::training_neighbor_sampler_)
        .def_readwrite("evaluation_neighbor_sampler", &DataLoader::evaluation_neighbor_sampler_)
        .def_readwrite("training_negative_sampler", &DataLoader::training_negative_sampler_)
        .def_readwrite("evaluation_negative_sampler", &DataLoader::evaluation_negative_sampler_)

        .def(py::init([](shared_ptr<GraphModelStorage> graph_storage, std::string learning_task, shared_ptr<TrainingConfig> training_config,
                         shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config) {
                 LearningTask task = getLearningTask(learning_task);
                 return std::make_shared<DataLoader>(graph_storage, task, training_config, evaluation_config, encoder_config);
             }),
             py::arg("graph_storage"), py::arg("learning_task"), py::arg("training_config"), py::arg("evaluation_config"), py::arg("encoder_config"))

        .def(py::init([](shared_ptr<GraphModelStorage> graph_storage, std::string learning_task, int batch_size, shared_ptr<NegativeSampler> neg_sampler,
                         shared_ptr<NeighborSampler> nbr_sampler, bool train) {
                 LearningTask task = getLearningTask(learning_task);

                 if (task == LearningTask::LINK_PREDICTION) {
                     if (train) {
                         graph_storage->storage_ptrs_.train_edges = graph_storage->storage_ptrs_.edges;
                     } else {
                         graph_storage->storage_ptrs_.test_edges = graph_storage->storage_ptrs_.edges;
                     }
                 } else if (task == LearningTask::NODE_CLASSIFICATION) {
                     if (graph_storage->storage_ptrs_.nodes == nullptr) {
                         throw MariusRuntimeException("Node ids must be provided for node classification");
                     }

                     if (train) {
                         if (graph_storage->storage_ptrs_.node_labels == nullptr) {
                             throw MariusRuntimeException("Labels for the nodes must be provided when training with node classification");
                         }
                         graph_storage->storage_ptrs_.train_nodes = graph_storage->storage_ptrs_.nodes;
                     } else {
                         graph_storage->storage_ptrs_.test_nodes = graph_storage->storage_ptrs_.nodes;
                     }
                 }

                 return std::make_shared<DataLoader>(graph_storage, task, batch_size, neg_sampler, nbr_sampler, train);
             }),
             py::arg("graph_storage"), py::arg("learning_task"), py::arg("batch_size") = 1000, py::arg("neg_sampler") = nullptr,
             py::arg("nbr_sampler") = nullptr, py::arg("train") = false)

        .def(py::init([](torch::optional<torch::Tensor> edges, std::string learning_task, torch::optional<torch::Tensor> nodes,
                         torch::optional<torch::Tensor> node_features, torch::optional<torch::Tensor> node_embeddings,
                         torch::optional<torch::Tensor> node_optimizer_state, torch::optional<torch::Tensor> node_labels,
                         torch::optional<torch::Tensor> train_edges, int batch_size, shared_ptr<NegativeSampler> neg_sampler,
                         shared_ptr<NeighborSampler> nbr_sampler, std::vector<torch::Tensor> filter_edges, bool train) {
                 shared_ptr<Storage> edges_s = nullptr;
                 shared_ptr<Storage> nodes_s = nullptr;
                 shared_ptr<Storage> node_features_s = nullptr;
                 shared_ptr<Storage> node_embeddings_s = nullptr;
                 shared_ptr<Storage> node_optimizer_state_s = nullptr;
                 shared_ptr<Storage> node_labels_s = nullptr;

                 LearningTask task = getLearningTask(learning_task);

                 if (edges.has_value()) {
                     edges_s = std::make_shared<InMemory>(edges.value());
                 } else {
                     throw UndefinedTensorException();
                 }

                 if (nodes.has_value()) {
                     nodes_s = std::make_shared<InMemory>(nodes.value());
                 } else {
                     if (task == LearningTask::NODE_CLASSIFICATION) {
                         throw MariusRuntimeException("Tensor of node ids must be provided for node classification");
                     }
                 }

                 if (node_features.has_value()) {
                     node_features_s = std::make_shared<InMemory>(node_features.value());
                 }

                 if (node_embeddings.has_value()) {
                     node_embeddings_s = std::make_shared<InMemory>(node_embeddings.value());
                 }

                 if (node_optimizer_state.has_value()) {
                     node_optimizer_state_s = std::make_shared<InMemory>(node_optimizer_state.value());
                 } else {
                     if (train && node_embeddings_s != nullptr) {
                         OptimizerState emb_state = torch::zeros_like(node_embeddings.value());
                         node_optimizer_state_s = std::make_shared<InMemory>(emb_state);
                     }
                 }

                 if (node_labels.has_value()) {
                     node_labels_s = std::make_shared<InMemory>(node_labels.value());
                 } else {
                     if (task == LearningTask::NODE_CLASSIFICATION && train) {
                         throw MariusRuntimeException("Labels for the nodes must be provided when training with node classification");
                     }
                 }

                 GraphModelStoragePtrs ptrs;
                 ptrs.edges = edges_s;
                 ptrs.nodes = nodes_s;
                 ptrs.node_features = node_features_s;
                 ptrs.node_embeddings = node_embeddings_s;
                 ptrs.node_optimizer_state = node_optimizer_state_s;
                 ptrs.node_labels = node_labels_s;

                 for (auto f_edges : filter_edges) {
                     ptrs.filter_edges.emplace_back(std::make_shared<InMemory>(f_edges));
                 }

                 if (task == LearningTask::LINK_PREDICTION) {
                     if (train) {
                         ptrs.train_edges = ptrs.edges;
                     } else {
                         ptrs.test_edges = ptrs.edges;
                         if (train_edges.has_value()) {
                             ptrs.train_edges = std::make_shared<InMemory>(train_edges.value());
                         }
                     }
                 } else if (task == LearningTask::NODE_CLASSIFICATION) {
                     if (train) {
                         ptrs.train_nodes = ptrs.nodes;
                     } else {
                         ptrs.test_nodes = ptrs.nodes;
                     }
                 }

                 auto gms = std::make_shared<GraphModelStorage>(ptrs, false);
                 return std::make_shared<DataLoader>(gms, task, batch_size, neg_sampler, nbr_sampler, train);
             }),
             py::arg("edges"), py::arg("learning_task"), py::arg("nodes") = nullptr, py::arg("node_features") = nullptr, py::arg("node_embeddings") = nullptr,
             py::arg("node_optim_state") = nullptr, py::arg("node_labels") = nullptr, py::arg("train_edges") = nullptr, py::arg("batch_size") = 1000,
             py::arg("neg_sampler") = nullptr, py::arg("nbr_sampler") = nullptr, py::arg("filter_edges") = std::vector<torch::Tensor>(),
             py::arg("train") = false)

        .def("setBufferOrdering", &DataLoader::setBufferOrdering)
        .def("setActiveEdges", &DataLoader::setActiveEdges)
        .def("setActiveNodes", &DataLoader::setActiveNodes)
        .def("initializeBatches", &DataLoader::initializeBatches, py::arg("prepare_encode") = false)
        .def("clearBatches", &DataLoader::clearBatches)
        .def("hasNextBatch", &DataLoader::hasNextBatch)
        .def("getNextBatch", &DataLoader::getNextBatch, py::return_value_policy::reference)
        .def("finishedBatch", &DataLoader::finishedBatch)
        .def("getBatch", &DataLoader::getBatch, py::arg("device") = py::none(), py::arg("perform_map") = false, py::arg("worker_id") = 0,
             py::return_value_policy::reference)
        .def("edgeSample", &DataLoader::edgeSample, py::arg("batch"), py::arg("worker_id") = 0)
        .def("nodeSample", &DataLoader::nodeSample, py::arg("batch"), py::arg("worker_id") = 0)
        .def("loadCPUParameters", &DataLoader::loadCPUParameters, py::arg("batch"))
        .def("loadGPUParameters", &DataLoader::loadGPUParameters, py::arg("batch"))
        .def("updateEmbeddings", &DataLoader::updateEmbeddings, py::arg("batch"), py::arg("gpu") = false)
        .def("nextEpoch", &DataLoader::nextEpoch)
        .def("loadStorage", &DataLoader::loadStorage)
        .def("epochComplete", &DataLoader::epochComplete)
        .def("unloadStorage", &DataLoader::unloadStorage, py::arg("write") = false)
        .def("getNumEdges", &DataLoader::getNumEdges)
        .def("getEpochsProcessed", &DataLoader::getEpochsProcessed)
        .def("getBatchesProcessed", &DataLoader::getBatchesProcessed)
        .def("isTrain", &DataLoader::isTrain)
        .def("setTrainSet", &DataLoader::setTrainSet)
        .def("setValidationSet", &DataLoader::setValidationSet)
        .def("setTestSet", &DataLoader::setTestSet);
}