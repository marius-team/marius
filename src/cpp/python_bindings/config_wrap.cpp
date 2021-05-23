#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <config.h>

namespace py = pybind11;

void init_config(py::module &m) {

    // GeneralOptions class
    py::class_<GeneralOptions>(m, "GeneralOptions")
        .def_readwrite("device", &GeneralOptions::device) // cast to pytorch type?
        .def_readwrite("random_seed", &GeneralOptions::random_seed)
        .def_readwrite("gpu_ids", &GeneralOptions::gpu_ids)
        .def_readwrite("num_train", &GeneralOptions::num_train)
        .def_readwrite("num_valid", &GeneralOptions::num_valid)
        .def_readwrite("num_test", &GeneralOptions::num_test)
        .def_readwrite("num_nodes", &GeneralOptions::num_nodes)
        .def_readwrite("num_relations", &GeneralOptions::num_relations)
        .def_readwrite("experiment_name", &GeneralOptions::experiment_name);

    // ModelOptions class
    py::class_<ModelOptions>(m, "ModelOptions")
        .def_readwrite("scale_factor", &ModelOptions::scale_factor)
        .def_readwrite("initialization_distribution", &ModelOptions::initialization_distribution)
        .def_readwrite("embedding_size", &ModelOptions::embedding_size)
        .def_readwrite("encoder_model", &ModelOptions::encoder_model)
        .def_readwrite("decoder_model", &ModelOptions::decoder_model)
        .def_readwrite("comparator", &ModelOptions::comparator)
        .def_readwrite("relation_operator", &ModelOptions::relation_operator);

     // StorageOptions class
    py::class_<StorageOptions>(m, "StorageOptions")
        .def_readwrite("edges", &StorageOptions::edges)
        .def_readwrite("reinitialize_edges", &StorageOptions::reinitialize_edges)
        .def_readwrite("remove_preprocessed", &StorageOptions::remove_preprocessed)
        .def_readwrite("shuffle_input_edges", &StorageOptions::shuffle_input_edges)
        .def_readwrite("edges_dtype", &StorageOptions::edges_dtype) // cast?
        .def_readwrite("embeddings", &StorageOptions::embeddings)
        .def_readwrite("reinitialize_embeddings", &StorageOptions::reinitialize_embeddings)
        .def_readwrite("relations", &StorageOptions::relations)
        .def_readwrite("embeddings_dtype", &StorageOptions::embeddings_dtype)
        .def_readwrite("edge_bucket_ordering", &StorageOptions::edge_bucket_ordering)
        .def_readwrite("num_partitions", &StorageOptions::num_partitions)
        .def_readwrite("buffer_capacity", &StorageOptions::buffer_capacity)
        .def_readwrite("prefetching", &StorageOptions::prefetching)
        .def_readwrite("conserve_memory", &StorageOptions::conserve_memory);

    // TrainingOptions class
    py::class_<TrainingOptions>(m, "TrainingOptions")
        .def_readwrite("batch_size", &TrainingOptions::batch_size)
        .def_readwrite("number_of_chunks", &TrainingOptions::number_of_chunks)
        .def_readwrite("negatives", &TrainingOptions::negatives)
        .def_readwrite("degree_fraction", &TrainingOptions::degree_fraction)
        .def_readwrite("negative_sampling_access", &TrainingOptions::negative_sampling_access)
        .def_readwrite("learning_rate", &TrainingOptions::learning_rate)
        .def_readwrite("regularization_coef", &TrainingOptions::regularization_coef)
        .def_readwrite("regularization_norm", &TrainingOptions::regularization_norm)
        .def_readwrite("optimizer_type", &TrainingOptions::optimizer_type)
        .def_readwrite("average_gradients", &TrainingOptions::average_gradients)
        .def_readwrite("synchronous", &TrainingOptions::synchronous)
        .def_readwrite("num_epochs", &TrainingOptions::num_epochs)
        .def_readwrite("checkpoint_interval", &TrainingOptions::checkpoint_interval)
        .def_readwrite("shuffle_interval", &TrainingOptions::shuffle_interval);

    // LossOptions class
    py::class_<LossOptions>(m, "LossOptions")
        .def_readwrite("loss_function_type", &LossOptions::loss_function_type)
        .def_readwrite("margin", &LossOptions::margin);

    // TrainingPipelineOptions class
    py::class_<TrainingPipelineOptions>(m, "TrainingPipelineOptions")
        .def_readwrite("max_batches_in_flight", &TrainingPipelineOptions::max_batches_in_flight)
        .def_readwrite("update_in_flight", &TrainingPipelineOptions::update_in_flight)
        .def_readwrite("embeddings_host_queue_size", &TrainingPipelineOptions::embeddings_host_queue_size)
        .def_readwrite("embeddings_device_queue_size", &TrainingPipelineOptions::embeddings_device_queue_size)
        .def_readwrite("gradients_host_queue_size", &TrainingPipelineOptions::gradients_host_queue_size)
        .def_readwrite("gradients_device_queue_size", &TrainingPipelineOptions::gradients_device_queue_size)
        .def_readwrite("num_embedding_loader_threads", &TrainingPipelineOptions::num_embedding_loader_threads)
        .def_readwrite("num_embedding_transfer_threads", &TrainingPipelineOptions::num_embedding_transfer_threads)
        .def_readwrite("num_compute_threads", &TrainingPipelineOptions::num_compute_threads)
        .def_readwrite("num_gradient_transfer_threads", &TrainingPipelineOptions::num_gradient_transfer_threads)
        .def_readwrite("num_embedding_update_threads", &TrainingPipelineOptions::num_embedding_update_threads);

    // EvaluationOptions class
    py::class_<EvaluationOptions>(m, "EvaluationOptions")
        .def_readwrite("batch_size", &EvaluationOptions::batch_size)
        .def_readwrite("number_of_chunks", &EvaluationOptions::number_of_chunks)
        .def_readwrite("negatives", &EvaluationOptions::negatives)
        .def_readwrite("degree_fraction", &EvaluationOptions::degree_fraction)
        .def_readwrite("negative_sampling_access", &EvaluationOptions::negative_sampling_access)
        .def_readwrite("epochs_per_eval", &EvaluationOptions::epochs_per_eval)
        .def_readwrite("synchronous", &EvaluationOptions::synchronous)
        .def_readwrite("filtered_evaluation", &EvaluationOptions::filtered_evaluation)
        .def_readwrite("checkpoint_to_eval", &EvaluationOptions::checkpoint_to_eval);

    // EvaluationPipelineOptions class
    py::class_<EvaluationPipelineOptions>(m, "EvaluationPipelineOptions")
        .def_readwrite("max_batches_in_flight", &EvaluationPipelineOptions::max_batches_in_flight)
        .def_readwrite("embeddings_host_queue_size", &EvaluationPipelineOptions::embeddings_host_queue_size)
        .def_readwrite("embeddings_device_queue_size", &EvaluationPipelineOptions::embeddings_device_queue_size)
        .def_readwrite("num_embedding_loader_threads", &EvaluationPipelineOptions::num_embedding_loader_threads)
        .def_readwrite("num_embedding_transfer_threads", &EvaluationPipelineOptions::num_embedding_transfer_threads)
        .def_readwrite("num_evaluate_threads", &EvaluationPipelineOptions::num_evaluate_threads);

    // PathOptions class
    py::class_<PathOptions>(m, "PathOptions")
        .def_readwrite("train_edges", &PathOptions::train_edges)
        .def_readwrite("train_edges_partitions", &PathOptions::train_edges_partitions)
        .def_readwrite("validation_edges", &PathOptions::validation_edges)
        .def_readwrite("validation_edges_partitions", &PathOptions::validation_edges_partitions)
        .def_readwrite("test_edges", &PathOptions::test_edges)
        .def_readwrite("test_edges_partitions", &PathOptions::test_edges_partitions)
        .def_readwrite("node_labels", &PathOptions::node_labels)
        .def_readwrite("relation_labels", &PathOptions::relation_labels)
        .def_readwrite("node_ids", &PathOptions::node_ids)
        .def_readwrite("relations_ids", &PathOptions::relations_ids)
        .def_readwrite("custom_ordering", &PathOptions::custom_ordering)
        .def_readwrite("base_directory", &PathOptions::base_directory)
        .def_readwrite("experiment_directory", &PathOptions::experiment_directory);

    py::class_<ReportingOptions>(m, "ReportingOptions")
        .def_readwrite("logs_per_epoch", &ReportingOptions::logs_per_epoch)
        .def_readwrite("log_level", &ReportingOptions::log_level); // cast?

    // MariusOptions class
    py::class_<MariusOptions>(m, "MariusOptions")
        .def_readwrite("general", &MariusOptions::general)
        .def_readwrite("model", &MariusOptions::model)
        .def_readwrite("storage", &MariusOptions::storage)
        .def_readwrite("training", &MariusOptions::training)
        .def_readwrite("training_pipeline", &MariusOptions::training_pipeline)
        .def_readwrite("evaluation", &MariusOptions::evaluation)
        .def_readwrite("evaluation_pipeline", &MariusOptions::evaluation_pipeline)
        .def_readwrite("path", &MariusOptions::path)
        .def_readwrite("reporting", &MariusOptions::reporting);


    m.def("parseConfig", [](string config_path) {
	// hacky, fix this
        char *argv[2];

        char *name = strdup(std::string("marius").c_str());
        char *config = strdup(config_path.c_str());
        argv[0] = name;
        argv[1] = config;
	    marius_options = parseConfig(2, argv);
        return marius_options;
    }, py::arg("config_path"), py::return_value_policy::reference);
}
