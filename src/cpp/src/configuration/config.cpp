//
// Created by Jason Mohoney on 10/8/21.
//

#include "configuration/config.h"

#include <common/pybind_headers.h>
#include <stdlib.h>

bool check_missing(pyobj python_object) {
    bool missing = false;
    try {
        string string_val = pybind11::cast<string>(python_object);

        if (string_val == MISSING_STR) {
            missing = true;
        }
    } catch (pybind11::cast_error) {
    }

    return missing;
}

template <typename T>
T cast_helper(pyobj python_object) {
    bool missing = check_missing(python_object);

    if (missing) {
        T default_value;
        return default_value;
    } else {
        return pybind11::cast<T>(python_object);
    }
}

shared_ptr<NeighborSamplingConfig> initNeighborSamplingConfig(pyobj python_object) {
    shared_ptr<NeighborSamplingConfig> ret_config = std::make_shared<NeighborSamplingConfig>();

    ret_config->type = getNeighborSamplingLayer(cast_helper<string>(python_object.attr("type")));

    pyobj py_options = python_object.attr("options");

    if (ret_config->type == NeighborSamplingLayer::UNIFORM) {
        auto uniform_options = std::make_shared<UniformSamplingOptions>();
        uniform_options->max_neighbors = cast_helper<int>(py_options.attr("max_neighbors"));
        ret_config->options = uniform_options;
    } else if (ret_config->type == NeighborSamplingLayer::DROPOUT) {
        auto dropout_options = std::make_shared<DropoutSamplingOptions>();
        dropout_options->rate = cast_helper<float>(py_options.attr("rate"));
        ret_config->options = dropout_options;
    } else {
        auto options = std::make_shared<NeighborSamplingOptions>();
        ret_config->options = options;
    }

    ret_config->use_hashmap_sets = cast_helper<bool>(python_object.attr("use_hashmap_sets"));

    return ret_config;
}

shared_ptr<InitConfig> initInitConfig(pyobj python_object) {
    shared_ptr<InitConfig> ret_config = std::make_shared<InitConfig>();

    ret_config->type = getInitDistribution(cast_helper<string>(python_object.attr("type")));

    pyobj py_options = python_object.attr("options");

    if (ret_config->type == InitDistribution::CONSTANT) {
        auto constant_options = std::make_shared<ConstantInitOptions>();
        constant_options->constant = cast_helper<float>(py_options.attr("constant"));
        ret_config->options = constant_options;
    } else if (ret_config->type == InitDistribution::UNIFORM) {
        auto uniform_options = std::make_shared<UniformInitOptions>();
        uniform_options->scale_factor = cast_helper<float>(py_options.attr("scale_factor"));
        ret_config->options = uniform_options;
    } else if (ret_config->type == InitDistribution::NORMAL) {
        auto normal_options = std::make_shared<NormalInitOptions>();
        normal_options->mean = cast_helper<float>(py_options.attr("mean"));
        normal_options->std = cast_helper<float>(py_options.attr("std"));
        ret_config->options = normal_options;
    } else {
        auto options = std::make_shared<InitOptions>();
        ret_config->options = options;
    }

    return ret_config;
}

shared_ptr<OptimizerConfig> initOptimizerConfig(pyobj python_config) {
    shared_ptr<OptimizerConfig> ret_config = std::make_shared<OptimizerConfig>();

    if (check_missing(python_config)) {
        return nullptr;
    }

    ret_config->type = getOptimizerType(cast_helper<string>(python_config.attr("type")));

    if (ret_config->type == OptimizerType::DEFAULT) {
        return nullptr;
    }

    pyobj py_options = python_config.attr("options");

    if (ret_config->type == OptimizerType::ADAGRAD) {
        auto adagrad_options = std::make_shared<AdagradOptions>();
        adagrad_options->weight_decay = cast_helper<float>(py_options.attr("weight_decay"));
        adagrad_options->lr_decay = cast_helper<float>(py_options.attr("lr_decay"));
        adagrad_options->init_value = cast_helper<float>(py_options.attr("init_value"));
        adagrad_options->eps = cast_helper<float>(py_options.attr("eps"));
        ret_config->options = adagrad_options;
    } else if (ret_config->type == OptimizerType::ADAM) {
        auto adam_options = std::make_shared<AdamOptions>();
        adam_options->weight_decay = cast_helper<float>(py_options.attr("weight_decay"));
        adam_options->amsgrad = cast_helper<bool>(py_options.attr("amsgrad"));
        adam_options->beta_1 = cast_helper<float>(py_options.attr("beta_1"));
        adam_options->beta_2 = cast_helper<float>(py_options.attr("beta_2"));
        adam_options->eps = cast_helper<float>(py_options.attr("eps"));
        ret_config->options = adam_options;
    } else {
        auto options = std::make_shared<OptimizerOptions>();
        ret_config->options = options;
    }

    ret_config->options->learning_rate = cast_helper<float>(py_options.attr("learning_rate"));

    return ret_config;
}

shared_ptr<DatasetConfig> initDatasetConfig(pyobj python_config) {
    shared_ptr<DatasetConfig> ret_config = std::make_shared<DatasetConfig>();

    ret_config->dataset_dir = cast_helper<string>(python_config.attr("dataset_dir"));
    ret_config->num_train = cast_helper<int64_t>(python_config.attr("num_train"));
    ret_config->num_valid = cast_helper<int64_t>(python_config.attr("num_valid"));
    ret_config->num_test = cast_helper<int64_t>(python_config.attr("num_test"));
    ret_config->num_edges = cast_helper<int64_t>(python_config.attr("num_edges"));
    ret_config->num_nodes = cast_helper<int64_t>(python_config.attr("num_nodes"));
    ret_config->num_relations = cast_helper<int64_t>(python_config.attr("num_relations"));
    ret_config->node_feature_dim = cast_helper<int>(python_config.attr("node_feature_dim"));
    ret_config->rel_feature_dim = cast_helper<int>(python_config.attr("rel_feature_dim"));
    ret_config->num_classes = cast_helper<int>(python_config.attr("num_classes"));

    return ret_config;
}

shared_ptr<LayerConfig> initLayerConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<LayerConfig> ret_config = std::make_shared<LayerConfig>();

    ret_config->type = getLayerType(cast_helper<string>(python_config.attr("type")));

    if (ret_config->type == LayerType::EMBEDDING) {
        ret_config->options = nullptr;
        ret_config->input_dim = -1;
        ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
        ret_config->init = initInitConfig(python_config.attr("init"));
        ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));
        ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
        ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
        ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));
    } else if (ret_config->type == LayerType::FEATURE) {
        ret_config->options = nullptr;
        ret_config->input_dim = -1;
        ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
        ret_config->init = nullptr;
        ret_config->optimizer = nullptr;
        ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
        ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
        ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));
    } else if (ret_config->type == LayerType::GNN) {
        pyobj py_options = python_config.attr("options");
        auto options = std::make_shared<GNNLayerOptions>();
        options->type = getGNNLayerType(cast_helper<string>(py_options.attr("type")));
        ret_config->input_dim = cast_helper<int>(python_config.attr("input_dim"));
        ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
        ret_config->init = initInitConfig(python_config.attr("init"));
        ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));
        ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
        ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
        ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));

        if (options->type == GNNLayerType::GRAPH_SAGE) {
            auto graph_sage_options = std::make_shared<GraphSageLayerOptions>();
            graph_sage_options->type = GNNLayerType::GRAPH_SAGE;
            graph_sage_options->aggregator = getGraphSageAggregator(cast_helper<string>(py_options.attr("aggregator")));
            ret_config->options = graph_sage_options;
        } else if (options->type == GNNLayerType::GAT) {
            auto gat_options = std::make_shared<GATLayerOptions>();
            gat_options->type = GNNLayerType::GAT;
            gat_options->num_heads = cast_helper<int>(py_options.attr("num_heads"));
            gat_options->negative_slope = cast_helper<float>(py_options.attr("negative_slope"));
            gat_options->average_heads = cast_helper<bool>(py_options.attr("average_heads"));
            gat_options->input_dropout = cast_helper<float>(py_options.attr("input_dropout"));
            gat_options->attention_dropout = cast_helper<float>(py_options.attr("attention_dropout"));
            ret_config->options = gat_options;
        } else {
            ret_config->options = std::make_shared<GNNLayerOptions>();
        }

    } else if (ret_config->type == LayerType::DENSE) {
        pyobj py_options = python_config.attr("options");
        auto options = std::make_shared<DenseLayerOptions>();
        options->type = getDenseLayerType(cast_helper<string>(py_options.attr("type")));
        ret_config->options = options;
        ret_config->input_dim = cast_helper<int>(python_config.attr("input_dim"));
        ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
        ret_config->init = initInitConfig(python_config.attr("init"));
        ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));
        ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
        ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
        ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));
    } else if (ret_config->type == LayerType::REDUCTION) {
        pyobj py_options = python_config.attr("options");
        auto options = std::make_shared<ReductionLayerOptions>();
        options->type = getReductionLayerType(cast_helper<string>(py_options.attr("type")));
        ret_config->options = options;
        ret_config->input_dim = cast_helper<int>(python_config.attr("input_dim"));
        ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
        ret_config->init = initInitConfig(python_config.attr("init"));
        ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));
        ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
        ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
        ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));
    }
    return ret_config;
}

shared_ptr<EncoderConfig> initEncoderConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<EncoderConfig> ret_config = std::make_shared<EncoderConfig>();

    pybind11::list stage_python_obj = cast_helper<pybind11::list>(python_config.attr("layers"));
    pybind11::list train_sample_python_obj = cast_helper<pybind11::list>(python_config.attr("train_neighbor_sampling"));
    pybind11::list eval_sample_python_obj = cast_helper<pybind11::list>(python_config.attr("eval_neighbor_sampling"));

    auto layer_vec = std::vector<std::vector<shared_ptr<LayerConfig>>>();
    auto train_sample_vec = std::vector<shared_ptr<NeighborSamplingConfig>>();
    auto eval_sample_vec = std::vector<shared_ptr<NeighborSamplingConfig>>();

    for (auto py_stage : stage_python_obj) {
        pybind11::list stage_obj = cast_helper<pybind11::list>(pybind11::reinterpret_borrow<pyobj>(py_stage));

        auto stage_vec = std::vector<shared_ptr<LayerConfig>>();

        for (auto py_layer : stage_obj) {
            pyobj layer_object = pybind11::reinterpret_borrow<pyobj>(py_layer);
            stage_vec.emplace_back(initLayerConfig(layer_object));
        }
        layer_vec.emplace_back(stage_vec);
    }

    for (auto py_layer : train_sample_python_obj) {
        pyobj layer_object = pybind11::reinterpret_borrow<pyobj>(py_layer);
        train_sample_vec.emplace_back(initNeighborSamplingConfig(layer_object));
    }

    for (auto py_layer : eval_sample_python_obj) {
        pyobj layer_object = pybind11::reinterpret_borrow<pyobj>(py_layer);
        eval_sample_vec.emplace_back(initNeighborSamplingConfig(layer_object));
    }

    ret_config->layers = layer_vec;
    ret_config->train_neighbor_sampling = train_sample_vec;
    ret_config->eval_neighbor_sampling = eval_sample_vec;
    ret_config->use_incoming_nbrs = cast_helper<bool>(python_config.attr("use_incoming_nbrs"));
    ret_config->use_outgoing_nbrs = cast_helper<bool>(python_config.attr("use_outgoing_nbrs"));

    return ret_config;
}

shared_ptr<DecoderConfig> initDecoderConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<DecoderConfig> ret_config = std::make_shared<DecoderConfig>();

    ret_config->type = getDecoderType(cast_helper<string>(python_config.attr("type")));
    ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));

    if (ret_config->type != DecoderType::NODE) {
        pyobj py_options = python_config.attr("options");
        auto options = std::make_shared<EdgeDecoderOptions>();
        options->inverse_edges = cast_helper<bool>(py_options.attr("inverse_edges"));
        options->edge_decoder_method = getEdgeDecoderMethod(cast_helper<string>(py_options.attr("edge_decoder_method")));
        ret_config->options = options;
    } else {
        auto options = std::make_shared<DecoderOptions>();
        ret_config->options = options;
    }

    return ret_config;
}

shared_ptr<LossConfig> initLossConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<LossConfig> ret_config = std::make_shared<LossConfig>();

    ret_config->type = getLossFunctionType(cast_helper<string>(python_config.attr("type")));

    pyobj py_options = python_config.attr("options");

    if (ret_config->type == LossFunctionType::RANKING) {
        auto ranking_options = std::make_shared<RankingLossOptions>();
        ranking_options->margin = cast_helper<float>(py_options.attr("margin"));
        ranking_options->loss_reduction = getLossReduction(cast_helper<string>(py_options.attr("reduction")));
        ret_config->options = ranking_options;
    } else {
        auto options = std::make_shared<LossOptions>();
        options->loss_reduction = getLossReduction(cast_helper<string>(py_options.attr("reduction")));
        ret_config->options = options;
    }

    return ret_config;
}

shared_ptr<StorageBackendConfig> initStorageBackendConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<StorageBackendConfig> ret_config = std::make_shared<StorageBackendConfig>();

    ret_config->type = getStorageBackend(cast_helper<string>(python_config.attr("type")));

    pyobj py_options = python_config.attr("options");

    if (ret_config->type == StorageBackend::PARTITION_BUFFER) {
        auto buffer_options = std::make_shared<PartitionBufferOptions>();
        buffer_options->num_partitions = cast_helper<int>(py_options.attr("num_partitions"));
        buffer_options->buffer_capacity = cast_helper<int>(py_options.attr("buffer_capacity"));
        buffer_options->prefetching = cast_helper<bool>(py_options.attr("prefetching"));
        buffer_options->fine_to_coarse_ratio = cast_helper<int>(py_options.attr("fine_to_coarse_ratio"));
        buffer_options->num_cache_partitions = cast_helper<int>(py_options.attr("num_cache_partitions"));
        buffer_options->edge_bucket_ordering = getEdgeBucketOrderingEnum(cast_helper<string>(py_options.attr("edge_bucket_ordering")));
        buffer_options->node_partition_ordering = getNodePartitionOrderingEnum(cast_helper<string>(py_options.attr("node_partition_ordering")));
        buffer_options->randomly_assign_edge_buckets = cast_helper<bool>(py_options.attr("randomly_assign_edge_buckets"));
        buffer_options->dtype = getDtype(cast_helper<string>(py_options.attr("dtype")));
        ret_config->options = buffer_options;
    } else {
        auto options = std::make_shared<StorageOptions>();
        options->dtype = getDtype(cast_helper<string>(py_options.attr("dtype")));
        ret_config->options = options;
    }

    return ret_config;
}

shared_ptr<NegativeSamplingConfig> initNegativeSamplingConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<NegativeSamplingConfig> ret_config = std::make_shared<NegativeSamplingConfig>();

    ret_config->filtered = cast_helper<bool>(python_config.attr("filtered"));
    if (!ret_config->filtered) {
        ret_config->negatives_per_positive = cast_helper<int>(python_config.attr("negatives_per_positive"));
        ret_config->num_chunks = cast_helper<int>(python_config.attr("num_chunks"));
        ret_config->degree_fraction = cast_helper<float>(python_config.attr("degree_fraction"));
        ret_config->local_filter_mode = getLocalFilterMode(cast_helper<std::string>(python_config.attr("local_filter_mode")));
    } else {
        ret_config->num_chunks = 1;
        ret_config->degree_fraction = 0.0;
        ret_config->negatives_per_positive = -1;  // This is set to the proper value by the graph_batcher
        ret_config->local_filter_mode = LocalFilterMode::DEG;
    }

    return ret_config;
}

shared_ptr<PipelineConfig> initPipelineConfig(pyobj python_config) {
    shared_ptr<PipelineConfig> ret_config = std::make_shared<PipelineConfig>();

    ret_config->sync = cast_helper<bool>(python_config.attr("sync"));
    if (!ret_config->sync) {
        ret_config->staleness_bound = cast_helper<int>(python_config.attr("staleness_bound"));
        ret_config->gpu_sync_interval = cast_helper<int>(python_config.attr("gpu_sync_interval"));
        ret_config->gpu_model_average = cast_helper<bool>(python_config.attr("gpu_model_average"));
        ret_config->batch_host_queue_size = cast_helper<int>(python_config.attr("batch_host_queue_size"));
        ret_config->batch_device_queue_size = cast_helper<int>(python_config.attr("batch_device_queue_size"));
        ret_config->gradients_device_queue_size = cast_helper<int>(python_config.attr("gradients_device_queue_size"));
        ret_config->gradients_host_queue_size = cast_helper<int>(python_config.attr("gradients_host_queue_size"));
        ret_config->batch_loader_threads = cast_helper<int>(python_config.attr("batch_loader_threads"));
        ret_config->batch_transfer_threads = cast_helper<int>(python_config.attr("batch_transfer_threads"));
        ret_config->compute_threads = cast_helper<int>(python_config.attr("compute_threads"));
        ret_config->gradient_transfer_threads = cast_helper<int>(python_config.attr("gradient_transfer_threads"));
        ret_config->gradient_update_threads = cast_helper<int>(python_config.attr("gradient_update_threads"));
    }

    return ret_config;
}

shared_ptr<CheckpointConfig> initCheckpointConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<CheckpointConfig> ret_config = std::make_shared<CheckpointConfig>();

    ret_config->save_best = cast_helper<bool>(python_config.attr("save_best"));
    ret_config->interval = cast_helper<int>(python_config.attr("interval"));
    ret_config->save_state = cast_helper<bool>(python_config.attr("save_state"));
    return ret_config;
}

shared_ptr<ModelConfig> initModelConfig(pyobj python_config) {
    shared_ptr<ModelConfig> ret_config = std::make_shared<ModelConfig>();

    ret_config->random_seed = cast_helper<int64_t>(python_config.attr("random_seed"));
    ret_config->learning_task = getLearningTask(cast_helper<std::string>(python_config.attr("learning_task")));
    ret_config->encoder = initEncoderConfig(python_config.attr("encoder"));
    ret_config->decoder = initDecoderConfig(python_config.attr("decoder"));
    ret_config->loss = initLossConfig(python_config.attr("loss"));
    ret_config->dense_optimizer = initOptimizerConfig(python_config.attr("dense_optimizer"));
    ret_config->sparse_optimizer = initOptimizerConfig(python_config.attr("sparse_optimizer"));

    return ret_config;
}

shared_ptr<StorageConfig> initStorageConfig(pyobj python_config) {
    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<StorageConfig> ret_config = std::make_shared<StorageConfig>();

    ret_config->device_type = torch::Device(cast_helper<string>(python_config.attr("device_type")));
    ret_config->edges = initStorageBackendConfig(python_config.attr("edges"));
    ret_config->nodes = initStorageBackendConfig(python_config.attr("nodes"));
    ret_config->embeddings = initStorageBackendConfig(python_config.attr("embeddings"));
    ret_config->features = initStorageBackendConfig(python_config.attr("features"));
    ret_config->dataset = initDatasetConfig(python_config.attr("dataset"));
    ret_config->prefetch = cast_helper<bool>(python_config.attr("prefetch"));
    ret_config->shuffle_input = cast_helper<bool>(python_config.attr("shuffle_input"));
    ret_config->model_dir = cast_helper<string>(python_config.attr("model_dir"));

    pybind11::list device_ids_pylist = cast_helper<pybind11::list>(python_config.attr("device_ids"));

    ret_config->device_ids = {};

    for (auto py_id : device_ids_pylist) {
        pyobj id_object = pybind11::reinterpret_borrow<pyobj>(py_id);
        ret_config->device_ids.emplace_back(cast_helper<int>(id_object));
    }

    ret_config->full_graph_evaluation = cast_helper<bool>(python_config.attr("full_graph_evaluation"));
    ret_config->export_encoded_nodes = cast_helper<bool>(python_config.attr("export_encoded_nodes"));

    ret_config->log_level = getLogLevel(cast_helper<string>(python_config.attr("log_level")));
    ret_config->train_edges_pre_sorted = cast_helper<bool>(python_config.attr("train_edges_pre_sorted"));
    return ret_config;
}

shared_ptr<TrainingConfig> initTrainingConfig(pyobj python_config) {
    shared_ptr<TrainingConfig> ret_config = std::make_shared<TrainingConfig>();

    ret_config->batch_size = cast_helper<int>(python_config.attr("batch_size"));
    ret_config->negative_sampling = initNegativeSamplingConfig(python_config.attr("negative_sampling"));
    ret_config->pipeline = initPipelineConfig(python_config.attr("pipeline"));
    ret_config->logs_per_epoch = cast_helper<int>(python_config.attr("logs_per_epoch"));
    ret_config->num_epochs = cast_helper<int>(python_config.attr("num_epochs"));
    ret_config->save_model = cast_helper<bool>(python_config.attr("save_model"));
    ret_config->checkpoint = initCheckpointConfig(python_config.attr("checkpoint"));
    ret_config->resume_training = cast_helper<bool>(python_config.attr("resume_training"));
    ret_config->resume_from_checkpoint = cast_helper<string>(python_config.attr("resume_from_checkpoint"));

    return ret_config;
}

shared_ptr<EvaluationConfig> initEvaluationConfig(pyobj python_config) {
    shared_ptr<EvaluationConfig> ret_config = std::make_shared<EvaluationConfig>();

    ret_config->batch_size = cast_helper<int>(python_config.attr("batch_size"));
    ret_config->negative_sampling = initNegativeSamplingConfig(python_config.attr("negative_sampling"));
    ret_config->pipeline = initPipelineConfig(python_config.attr("pipeline"));
    ret_config->epochs_per_eval = cast_helper<int>(python_config.attr("epochs_per_eval"));
    ret_config->checkpoint_dir = cast_helper<string>(python_config.attr("checkpoint_dir"));
    return ret_config;
}

shared_ptr<MariusConfig> initMariusConfig(pyobj python_config) {
    shared_ptr<MariusConfig> ret_config = std::make_shared<MariusConfig>();

    ret_config->model = initModelConfig(python_config.attr("model"));
    ret_config->storage = initStorageConfig(python_config.attr("storage"));
    ret_config->training = initTrainingConfig(python_config.attr("training"));
    ret_config->evaluation = initEvaluationConfig(python_config.attr("evaluation"));

    return ret_config;
}

shared_ptr<MariusConfig> loadConfig(string config_path, bool save) {
    string module_name = "marius.tools.configuration.marius_config";
    shared_ptr<MariusConfig> ret;
    if (Py_IsInitialized() != 0) {
        pyobj config_module = pybind11::module::import(module_name.c_str());
        pyobj python_config = config_module.attr("load_config")(config_path, save);

        ret = initMariusConfig(python_config);
    } else {
        setenv("MARIUS_NO_BINDINGS", "1", true);

        pybind11::scoped_interpreter guard{};

        pyobj config_module = pybind11::module::import(module_name.c_str());
        pyobj python_config = config_module.attr("load_config")(config_path, save);

        ret = initMariusConfig(python_config);
    }

    return ret;
}