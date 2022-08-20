#include <pybind11/pybind11.h>

#include "configuration/config.h"

namespace py = pybind11;

void init_config(py::module &m) {

    py::class_<NeighborSamplingConfig>(m, "NeighborSamplingConfig")
        .def_readwrite("type", &NeighborSamplingConfig::type)
        .def_readwrite("options", &NeighborSamplingConfig::options);

    py::class_<OptimizerConfig>(m, "OptimizerConfig")
        .def_readwrite("type", &OptimizerConfig::type)
        .def_readwrite("options", &OptimizerConfig::options);

    py::class_<InitConfig>(m, "InitConfig")
        .def_readwrite("type", &InitConfig::type)
        .def_readwrite("options", &InitConfig::options);

    py::class_<LossConfig>(m, "LossConfig")
        .def_readwrite("type", &LossConfig::type)
        .def_readwrite("options", &LossConfig::options);

    py::class_<EmbeddingsConfig>(m, "EmbeddingsConfig")
        .def_readwrite("dimension", &EmbeddingsConfig::dimension)
        .def_readwrite("init", &EmbeddingsConfig::init)
        .def_readwrite("optimizer", &EmbeddingsConfig::optimizer);

    py::class_<FeaturizerConfig>(m, "FeaturizerConfig")
        .def_readwrite("type", &FeaturizerConfig::type)
        .def_readwrite("options", &FeaturizerConfig::options)
        .def_readwrite("optimizer", &FeaturizerConfig::optimizer);

    py::class_<GNNLayerConfig>(m, "GNNLayerConfig")
        .def_readwrite("train_neighbor_sampling", &GNNLayerConfig::train_neighbor_sampling)
        .def_readwrite("eval_neighbor_sampling", &GNNLayerConfig::eval_neighbor_sampling)
        .def_readwrite("init", &GNNLayerConfig::init)
        .def_readwrite("type", &GNNLayerConfig::type)
        .def_readwrite("options", &GNNLayerConfig::options)
        .def_readwrite("activation", &GNNLayerConfig::activation)
        .def_readwrite("bias", &GNNLayerConfig::bias)
        .def_readwrite("bias_init", &GNNLayerConfig::bias_init);

    py::class_<EncoderConfig>(m, "EncoderConfig")
        .def_readwrite("input_dim", &EncoderConfig::input_dim)
        .def_readwrite("output_dim", &EncoderConfig::output_dim)
        .def_readwrite("layers", &EncoderConfig::layers)
        .def_readwrite("optimizer", &EncoderConfig::optimizer)
        .def_readwrite("use_incoming_nbrs", &EncoderConfig::use_incoming_nbrs)
        .def_readwrite("use_outgoing_nbrs", &EncoderConfig::use_outgoing_nbrs);

    py::class_<DecoderConfig>(m, "DecoderConfig")
        .def_readwrite("type", &DecoderConfig::type)
        .def_readwrite("options", &DecoderConfig::options)
        .def_readwrite("optimizer", &DecoderConfig::optimizer);

    py::class_<StorageBackendConfig>(m, "StorageBackendConfig")
        .def_readwrite("type", &StorageBackendConfig::type)
        .def_readwrite("options", &StorageBackendConfig::options);

    py::class_<DatasetConfig>(m, "DatasetConfig")
        .def_readwrite("base_directory", &DatasetConfig::base_directory)
        .def_readwrite("num_edges", &DatasetConfig::num_edges)
        .def_readwrite("num_nodes", &DatasetConfig::num_nodes)
        .def_readwrite("num_relations", &DatasetConfig::num_relations)
        .def_readwrite("num_train", &DatasetConfig::num_train)
        .def_readwrite("num_valid", &DatasetConfig::num_valid)
        .def_readwrite("num_test", &DatasetConfig::num_test)
        .def_readwrite("feature_dim", &DatasetConfig::feature_dim)
        .def_readwrite("num_classes", &DatasetConfig::num_classes);
    
    py::class_<NegativeSamplingConfig>(m, "NegativeSamplingConfig")
        .def_readwrite("num_chunks", &NegativeSamplingConfig::num_chunks)
        .def_readwrite("negatives_per_positive", &NegativeSamplingConfig::negatives_per_positive)
        .def_readwrite("degree_fraction", &NegativeSamplingConfig::degree_fraction)
        .def_readwrite("filtered", &NegativeSamplingConfig::filtered);

    py::class_<PipelineConfig>(m, "PipelineConfig")
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

    py::class_<ModelConfig, std::shared_ptr<ModelConfig>>(m, "ModelConfig")
        .def_readwrite("random_seed", &ModelConfig::random_seed)
        .def_readwrite("learning_task", &ModelConfig::learning_task)
        .def_readwrite("embeddings", &ModelConfig::embeddings)
        .def_readwrite("featurizer", &ModelConfig::featurizer)
        .def_readwrite("encoder", &ModelConfig::encoder)
        .def_readwrite("decoder", &ModelConfig::decoder)
        .def_readwrite("loss", &ModelConfig::loss);

    py::class_<StorageConfig>(m, "StorageConfig")
        .def_readwrite("device_type", &StorageConfig::device_type)
        .def_readwrite("device_ids", &StorageConfig::device_ids)
        .def_readwrite("dataset", &StorageConfig::dataset)
        .def_readwrite("edges", &StorageConfig::edges)
        .def_readwrite("embeddings", &StorageConfig::embeddings)
        .def_readwrite("features", &StorageConfig::features)
        .def_readwrite("shuffle_input", &StorageConfig::shuffle_input)
        .def_readwrite("full_graph_evaluation", &StorageConfig::full_graph_evaluation);

    py::class_<TrainingConfig, std::shared_ptr<TrainingConfig>>(m, "TrainingConfig")
        .def_readwrite("batch_size", &TrainingConfig::batch_size)
        .def_readwrite("negative_sampling", &TrainingConfig::negative_sampling)
        .def_readwrite("num_epochs", &TrainingConfig::num_epochs)
        .def_readwrite("pipeline", &TrainingConfig::pipeline)
        .def_readwrite("epochs_per_shuffle", &TrainingConfig::epochs_per_shuffle)
        .def_readwrite("logs_per_epoch", &TrainingConfig::logs_per_epoch);

    py::class_<EvaluationConfig, std::shared_ptr<EvaluationConfig>>(m, "EvaluationConfig")
        .def_readwrite("batch_size", &EvaluationConfig::batch_size)
        .def_readwrite("negative_sampling", &EvaluationConfig::negative_sampling)
        .def_readwrite("pipeline", &EvaluationConfig::pipeline)
        .def_readwrite("epochs_per_eval", &EvaluationConfig::epochs_per_eval)
        .def_readwrite("eval_checkpoint", &EvaluationConfig::eval_checkpoint)
        .def_readwrite("full_graph_evaluation", &EvaluationConfig::full_graph_evaluation);

    py::class_<MariusConfig, std::shared_ptr<MariusConfig>>(m, "MariusConfig")
        .def_readwrite("model", &MariusConfig::model)
        .def_readwrite("storage", &MariusConfig::storage)
        .def_readwrite("training", &MariusConfig::training)
        .def_readwrite("evaluation", &MariusConfig::evaluation);

    m.def("initConfig", &initConfig, py::arg("config_path") = "");
}