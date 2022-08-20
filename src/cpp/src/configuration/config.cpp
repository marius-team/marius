//
// Created by Jason Mohoney on 10/8/21.
//

#include "configuration/config.h"

bool check_missing(pyobj python_object) {
    bool missing = false;
    try {
        string string_val = pybind11::cast<string>(python_object);

        if (string_val == MISSING_STR) {
            missing = true;
        }
    } catch (pybind11::cast_error) {}

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

    ret_config->type = getOptimizerType(cast_helper<string>(python_config.attr("type")));

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

    ret_config->base_directory = cast_helper<string>(python_config.attr("base_directory"));
    ret_config->num_train = cast_helper<int64_t>(python_config.attr("num_train"));
    ret_config->num_valid = cast_helper<int64_t>(python_config.attr("num_valid"));
    ret_config->num_test = cast_helper<int64_t>(python_config.attr("num_test"));
    ret_config->num_edges = cast_helper<int64_t>(python_config.attr("num_edges"));
    ret_config->num_nodes = cast_helper<int64_t>(python_config.attr("num_nodes"));
    ret_config->num_relations = cast_helper<int64_t>(python_config.attr("num_relations"));
    ret_config->feature_dim = cast_helper<int>(python_config.attr("feature_dim"));
    ret_config->num_classes = cast_helper<int>(python_config.attr("num_classes"));

    return ret_config;
}

shared_ptr<EmbeddingsConfig> initEmbeddingsConfig(pyobj python_config) {

    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<EmbeddingsConfig> ret_config = std::make_shared<EmbeddingsConfig>();

    ret_config->dimension = cast_helper<int>(python_config.attr("dimension"));
    ret_config->init = initInitConfig(python_config.attr("init"));
    ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));

    return ret_config;
}

shared_ptr<FeaturizerConfig> initFeaturizerConfig(pyobj python_config) {

    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<FeaturizerConfig> ret_config = std::make_shared<FeaturizerConfig>();

    ret_config->type = getFeaturizerType(cast_helper<string>(python_config.attr("type")));
    ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));
    auto options = std::make_shared<FeaturizerOptions>();
    ret_config->options = options;

    return ret_config;
}

shared_ptr<GNNLayerConfig> initGNNLayerConfig(pyobj python_config) {
    shared_ptr<GNNLayerConfig> ret_config = std::make_shared<GNNLayerConfig>();

    ret_config->type = getGNNLayerType(cast_helper<string>(python_config.attr("type")));
    ret_config->train_neighbor_sampling = initNeighborSamplingConfig(python_config.attr("train_neighbor_sampling"));
    ret_config->eval_neighbor_sampling = initNeighborSamplingConfig(python_config.attr("eval_neighbor_sampling"));
    ret_config->activation = getActivationFunction(cast_helper<string>(python_config.attr("activation")));
    ret_config->bias = cast_helper<bool>(python_config.attr("bias"));
    ret_config->bias_init = initInitConfig(python_config.attr("bias_init"));
    ret_config->init = initInitConfig(python_config.attr("init"));

    pyobj py_options = python_config.attr("options");

    if (ret_config->type == GNNLayerType::GRAPH_SAGE) {
        auto graph_sage_options = std::make_shared<GraphSageLayerOptions>();
        graph_sage_options->aggregator = getGraphSageAggregator(cast_helper<string>(py_options.attr("aggregator")));
        ret_config->options = graph_sage_options;
    } else if (ret_config->type == GNNLayerType::GAT) {
        auto gat_options = std::make_shared<GATLayerOptions>();
        gat_options->num_heads = cast_helper<int>(py_options.attr("num_heads"));
        gat_options->negative_slope = cast_helper<float>(py_options.attr("negative_slope"));
        gat_options->average_heads = cast_helper<bool>(py_options.attr("average_heads"));
        gat_options->input_dropout = cast_helper<float>(py_options.attr("input_dropout"));
        gat_options->attention_dropout = cast_helper<float>(py_options.attr("attention_dropout"));
        ret_config->options = gat_options;
    } else {
        ret_config->options = std::make_shared<GNNLayerOptions>();
    }

    ret_config->options->input_dim = cast_helper<int>(py_options.attr("input_dim"));
    ret_config->options->output_dim = cast_helper<int>(py_options.attr("output_dim"));

    return ret_config;
}

shared_ptr<EncoderConfig> initEncoderConfig(pyobj python_config) {

    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<EncoderConfig> ret_config = std::make_shared<EncoderConfig>();

    ret_config->input_dim = cast_helper<int>(python_config.attr("input_dim"));
    ret_config->output_dim = cast_helper<int>(python_config.attr("output_dim"));
    ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));

    pybind11::list layer_python_obj = cast_helper<pybind11::list>(python_config.attr("layers"));

    auto layer_vec = std::vector<shared_ptr<GNNLayerConfig>>();

    for (auto py_layer : layer_python_obj) {
        pyobj layer_object = pybind11::reinterpret_borrow<pyobj>(py_layer);
        layer_vec.emplace_back(initGNNLayerConfig(layer_object));
    }

    ret_config->layers = layer_vec;

    ret_config->use_incoming_nbrs = cast_helper<bool>(python_config.attr("use_incoming_nbrs"));
    ret_config->use_outgoing_nbrs = cast_helper<bool>(python_config.attr("use_outgoing_nbrs"));
    ret_config->use_hashmap_sets = cast_helper<bool>(python_config.attr("use_hashmap_sets"));

    return ret_config;
}

shared_ptr<DecoderConfig> initDecoderConfig(pyobj python_config) {

    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<DecoderConfig> ret_config = std::make_shared<DecoderConfig>();

    ret_config->type = getDecoderType(cast_helper<string>(python_config.attr("type")));
    ret_config->optimizer = initOptimizerConfig(python_config.attr("optimizer"));

    pyobj py_options = python_config.attr("options");
    auto options = std::make_shared<DecoderOptions>();
    options->input_dim = cast_helper<int>(py_options.attr("input_dim"));
    options->inverse_edges = cast_helper<bool>(py_options.attr("inverse_edges"));
    ret_config->options = options;

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
    } else {
        ret_config->num_chunks = 1;
        ret_config->degree_fraction = 0.0;
        ret_config->negatives_per_positive = -1; // This is set to the proper value by the graph_batcher
    }

    return ret_config;
}

shared_ptr<PipelineConfig> initPipelineConfig(pyobj python_config) {
    shared_ptr<PipelineConfig> ret_config = std::make_shared<PipelineConfig>();

    ret_config->sync = cast_helper<bool>(python_config.attr("sync"));
    if (!ret_config->sync) {
        ret_config->staleness_bound = cast_helper<int>(python_config.attr("staleness_bound"));
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

shared_ptr<ModelConfig> initModelConfig(pyobj python_config) {
    shared_ptr<ModelConfig> ret_config = std::make_shared<ModelConfig>();

    ret_config->random_seed = cast_helper<int64_t>(python_config.attr("random_seed"));
    ret_config->learning_task = getLearningTask(cast_helper<std::string>(python_config.attr("learning_task")));
    ret_config->embeddings = initEmbeddingsConfig(python_config.attr("embeddings"));
    ret_config->featurizer = initFeaturizerConfig(python_config.attr("featurizer"));
    ret_config->encoder = initEncoderConfig(python_config.attr("encoder"));
    ret_config->decoder = initDecoderConfig(python_config.attr("decoder"));
    ret_config->loss = initLossConfig(python_config.attr("loss"));

    return ret_config;
}

shared_ptr<StorageConfig> initStorageConfig(pyobj python_config) {

    if (check_missing(python_config)) {
        return nullptr;
    }

    shared_ptr<StorageConfig> ret_config = std::make_shared<StorageConfig>();

    ret_config->device_type = torch::Device(cast_helper<string>(python_config.attr("device_type"))).type();
    ret_config->edges = initStorageBackendConfig(python_config.attr("edges"));
    ret_config->nodes = initStorageBackendConfig(python_config.attr("nodes"));
    ret_config->embeddings = initStorageBackendConfig(python_config.attr("embeddings"));
    ret_config->features = initStorageBackendConfig(python_config.attr("features"));
    ret_config->dataset = initDatasetConfig(python_config.attr("dataset"));
    ret_config->prefetch = cast_helper<bool>(python_config.attr("prefetch"));
    ret_config->shuffle_input = cast_helper<bool>(python_config.attr("shuffle_input"));

    pybind11::list device_ids_pylist = cast_helper<pybind11::list>(python_config.attr("device_ids"));

    ret_config->device_ids = {};

    for (auto py_id : device_ids_pylist) {
        pyobj id_object = pybind11::reinterpret_borrow<pyobj>(py_id);
        ret_config->device_ids.emplace_back(cast_helper<int>(id_object));
    }

    ret_config->full_graph_evaluation = cast_helper<bool>(python_config.attr("full_graph_evaluation"));


    return ret_config;
}

shared_ptr<TrainingConfig> initTrainingConfig(pyobj python_config) {
    shared_ptr<TrainingConfig> ret_config = std::make_shared<TrainingConfig>();

    ret_config->batch_size = cast_helper<int>(python_config.attr("batch_size"));
    ret_config->negative_sampling = initNegativeSamplingConfig(python_config.attr("negative_sampling"));
    ret_config->pipeline = initPipelineConfig(python_config.attr("pipeline"));
    ret_config->logs_per_epoch = cast_helper<int>(python_config.attr("logs_per_epoch"));
    ret_config->num_epochs = cast_helper<int>(python_config.attr("num_epochs"));

    return ret_config;
}

shared_ptr<EvaluationConfig> initEvaluationConfig(pyobj python_config) {
    shared_ptr<EvaluationConfig> ret_config = std::make_shared<EvaluationConfig>();

    ret_config->batch_size = cast_helper<int>(python_config.attr("batch_size"));
    ret_config->negative_sampling = initNegativeSamplingConfig(python_config.attr("negative_sampling"));
    ret_config->pipeline = initPipelineConfig(python_config.attr("pipeline"));
    ret_config->epochs_per_eval = cast_helper<int>(python_config.attr("epochs_per_eval"));
    ret_config->eval_checkpoint = cast_helper<int>(python_config.attr("eval_checkpoint"));
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

shared_ptr<MariusConfig> initConfig(string config_path) {

    shared_ptr<MariusConfig> ret;
    if (Py_IsInitialized() != 0) {
        string module_name = "marius.tools.configuration.marius_config";
        pyobj config_module = pybind11::module::import(module_name.c_str());
        pyobj python_config = config_module.attr("load_config")(config_path);

        ret = initMariusConfig(python_config);
    } else {
        pybind11::scoped_interpreter guard{};

        string module_name = "marius.tools.configuration.marius_config";
        pyobj config_module = pybind11::module::import(module_name.c_str());
        pyobj python_config = config_module.attr("load_config")(config_path);

        ret = initMariusConfig(python_config);
    }

    return ret;
}