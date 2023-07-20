#include "common/pybind_headers.h"
#include "configuration/config.h"

void init_config(py::module &m) {
    py::class_<NeighborSamplingConfig, std::shared_ptr<NeighborSamplingConfig>>(m, "NeighborSamplingConfig")
        .def(py::init<>())
        .def_readwrite("type", &NeighborSamplingConfig::type)
        .def_readwrite("options", &NeighborSamplingConfig::options)
        .def_readwrite("use_hashmap_sets", &NeighborSamplingConfig::use_hashmap_sets);

    py::class_<OptimizerConfig, std::shared_ptr<OptimizerConfig>>(m, "OptimizerConfig")
        .def(py::init<>())
        .def_readwrite("type", &OptimizerConfig::type)
        .def_readwrite("options", &OptimizerConfig::options);

    py::class_<InitConfig, std::shared_ptr<InitConfig>>(m, "InitConfig")
        .def(py::init<InitDistribution, std::shared_ptr<InitOptions>>(), py::arg("distribution"), py::arg("options"))
        .def_readwrite("type", &InitConfig::type)
        .def_readwrite("options", &InitConfig::options);

    py::class_<LossConfig, std::shared_ptr<LossConfig>>(m, "LossConfig")
        .def(py::init<>())
        .def_readwrite("type", &LossConfig::type)
        .def_readwrite("options", &LossConfig::options);

    py::class_<LayerConfig, std::shared_ptr<LayerConfig>>(m, "LayerConfig")
        .def(py::init<>())
        .def_readwrite("type", &LayerConfig::type)
        .def_readwrite("options", &LayerConfig::options)
        .def_readwrite("input_dim", &LayerConfig::input_dim)
        .def_readwrite("output_dim", &LayerConfig::output_dim)
        .def_readwrite("init", &LayerConfig::init)
        .def_readwrite("optimizer", &LayerConfig::optimizer)
        .def_readwrite("bias", &LayerConfig::bias)
        .def_readwrite("bias_init", &LayerConfig::bias_init)
        .def_readwrite("activation", &LayerConfig::activation);

    py::class_<EncoderConfig, std::shared_ptr<EncoderConfig>>(m, "EncoderConfig")
        .def(py::init<>())
        .def_readwrite("layers", &EncoderConfig::layers)
        .def_readwrite("train_neighbor_sampling", &EncoderConfig::train_neighbor_sampling)
        .def_readwrite("eval_neighbor_sampling", &EncoderConfig::eval_neighbor_sampling)
        .def_readwrite("use_incoming_nbrs", &EncoderConfig::use_incoming_nbrs)
        .def_readwrite("use_outgoing_nbrs", &EncoderConfig::use_outgoing_nbrs);

    py::class_<DecoderConfig, std::shared_ptr<DecoderConfig>>(m, "DecoderConfig")
        .def(py::init<>())
        .def_readwrite("type", &DecoderConfig::type)
        .def_readwrite("options", &DecoderConfig::options)
        .def_readwrite("optimizer", &DecoderConfig::optimizer);

    py::class_<StorageBackendConfig, std::shared_ptr<StorageBackendConfig>>(m, "StorageBackendConfig")
        .def(py::init<>())
        .def_readwrite("type", &StorageBackendConfig::type)
        .def_readwrite("options", &StorageBackendConfig::options);

    py::class_<DatasetConfig, std::shared_ptr<DatasetConfig>>(m, "DatasetConfig")
        .def(py::init<>())
        .def_readwrite("dataset_dir", &DatasetConfig::dataset_dir)
        .def_readwrite("num_edges", &DatasetConfig::num_edges)
        .def_readwrite("num_nodes", &DatasetConfig::num_nodes)
        .def_readwrite("num_relations", &DatasetConfig::num_relations)
        .def_readwrite("num_train", &DatasetConfig::num_train)
        .def_readwrite("num_valid", &DatasetConfig::num_valid)
        .def_readwrite("num_test", &DatasetConfig::num_test)
        .def_readwrite("node_feature_dim", &DatasetConfig::node_feature_dim)
        .def_readwrite("rel_feature_dim", &DatasetConfig::rel_feature_dim)
        .def_readwrite("num_classes", &DatasetConfig::num_classes);

    py::class_<NegativeSamplingConfig, std::shared_ptr<NegativeSamplingConfig>>(m, "NegativeSamplingConfig")
        .def(py::init<>())
        .def_readwrite("num_chunks", &NegativeSamplingConfig::num_chunks)
        .def_readwrite("negatives_per_positive", &NegativeSamplingConfig::negatives_per_positive)
        .def_readwrite("degree_fraction", &NegativeSamplingConfig::degree_fraction)
        .def_readwrite("filtered", &NegativeSamplingConfig::filtered);

    py::class_<PipelineConfig, std::shared_ptr<PipelineConfig>>(m, "PipelineConfig")
        .def(py::init<>())
        .def_readwrite("sync", &PipelineConfig::sync)
        .def_readwrite("staleness_bound", &PipelineConfig::staleness_bound)
        .def_readwrite("batch_host_queue_size", &PipelineConfig::batch_host_queue_size)
        .def_readwrite("batch_device_queue_size", &PipelineConfig::batch_device_queue_size)
        .def_readwrite("gradients_device_queue_size", &PipelineConfig::gradients_device_queue_size)
        .def_readwrite("gradients_host_queue_size", &PipelineConfig::gradients_host_queue_size)
        .def_readwrite("batch_loader_threads", &PipelineConfig::batch_loader_threads)
        .def_readwrite("batch_transfer_threads", &PipelineConfig::batch_transfer_threads)
        .def_readwrite("compute_threads", &PipelineConfig::compute_threads)
        .def_readwrite("gradient_transfer_threads", &PipelineConfig::gradient_transfer_threads)
        .def_readwrite("gradient_update_threads", &PipelineConfig::gradient_update_threads);

    py::class_<CheckpointConfig, std::shared_ptr<CheckpointConfig>>(m, "CheckpointConfig")
        .def(py::init<>())
        .def_readwrite("save_best", &CheckpointConfig::save_best)
        .def_readwrite("interval", &CheckpointConfig::interval)
        .def_readwrite("save_state", &CheckpointConfig::save_state);

    py::class_<ModelConfig, std::shared_ptr<ModelConfig>>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("random_seed", &ModelConfig::random_seed)
        .def_readwrite("learning_task", &ModelConfig::learning_task)
        .def_readwrite("encoder", &ModelConfig::encoder)
        .def_readwrite("decoder", &ModelConfig::decoder)
        .def_readwrite("loss", &ModelConfig::loss)
        .def_readwrite("dense_optimizer", &ModelConfig::dense_optimizer)
        .def_readwrite("sparse_optimizer", &ModelConfig::sparse_optimizer);

    py::class_<StorageConfig, std::shared_ptr<StorageConfig>>(m, "StorageConfig")
        .def(py::init<>())
        .def_readwrite("device_type", &StorageConfig::device_type)
        .def_readwrite("device_ids", &StorageConfig::device_ids)
        .def_readwrite("dataset", &StorageConfig::dataset)
        .def_readwrite("edges", &StorageConfig::edges)
        .def_readwrite("nodes", &StorageConfig::nodes)
        .def_readwrite("embeddings", &StorageConfig::embeddings)
        .def_readwrite("features", &StorageConfig::features)
        .def_readwrite("prefetch", &StorageConfig::prefetch)
        .def_readwrite("shuffle_input", &StorageConfig::shuffle_input)
        .def_readwrite("full_graph_evaluation", &StorageConfig::full_graph_evaluation)
        .def_readwrite("model_dir", &StorageConfig::model_dir)
        .def_readwrite("export_encoded_nodes", &StorageConfig::export_encoded_nodes);

    py::class_<TrainingConfig, std::shared_ptr<TrainingConfig>>(m, "TrainingConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &TrainingConfig::batch_size)
        .def_readwrite("negative_sampling", &TrainingConfig::negative_sampling)
        .def_readwrite("num_epochs", &TrainingConfig::num_epochs)
        .def_readwrite("pipeline", &TrainingConfig::pipeline)
        .def_readwrite("epochs_per_shuffle", &TrainingConfig::epochs_per_shuffle)
        .def_readwrite("logs_per_epoch", &TrainingConfig::logs_per_epoch)
        .def_readwrite("save_model", &TrainingConfig::save_model)
        .def_readwrite("checkpoint", &TrainingConfig::checkpoint)
        .def_readwrite("resume_training", &TrainingConfig::resume_training)
        .def_readwrite("resume_from_checkpoint", &TrainingConfig::resume_from_checkpoint);

    py::class_<EvaluationConfig, std::shared_ptr<EvaluationConfig>>(m, "EvaluationConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &EvaluationConfig::batch_size)
        .def_readwrite("negative_sampling", &EvaluationConfig::negative_sampling)
        .def_readwrite("pipeline", &EvaluationConfig::pipeline)
        .def_readwrite("epochs_per_eval", &EvaluationConfig::epochs_per_eval)
        .def_readwrite("full_graph_evaluation", &EvaluationConfig::full_graph_evaluation);

    py::class_<MariusConfig, std::shared_ptr<MariusConfig>>(m, "MariusConfig")
        .def(py::init<>())
        .def_readwrite("model", &MariusConfig::model)
        .def_readwrite("storage", &MariusConfig::storage)
        .def_readwrite("training", &MariusConfig::training)
        .def_readwrite("evaluation", &MariusConfig::evaluation);

    m.def("loadConfig", &loadConfig, py::arg("config_path"), py::arg("save") = false);
}