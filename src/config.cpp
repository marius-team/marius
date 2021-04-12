//
// Created by Jason Mohoney on 2/18/20.
//

#include <config.h>

MariusOptions marius_options = MariusOptions();
TimestampAllocator global_timestamp_allocator = TimestampAllocator();

MariusOptions parseConfig(string config_path, int64_t argc, char *argv[]) {
    // config_path validity check
    std::filesystem::path config_file_path = config_path;
    if (!std::filesystem::exists(config_file_path)){
        SPDLOG_ERROR("Unable to find configuration file: {}", config_path);
        exit(1);
    }

    // General options
    torch::DeviceType device;
    string s_device;
    string s_gpu_ids;
    std::vector<int> gpu_ids;
    int64_t rand_seed;
    int64_t num_train;
    int64_t num_valid;
    int64_t num_test;
    int64_t num_nodes;
    int64_t num_relations;
    string experiment_name;

    // Model options
    float scale_factor;
    InitializationDistribution initialization_distribution;
    string s_initialization_distribution;
    int embedding_size;
    EncoderModelType encoder_model;
    string s_encoder_model;
    DecoderModelType decoder_model;
    string s_decoder_model;

    // Storage options
    BackendType edges_backend_type;
    string s_edges_backend_type;
    bool reinit_edges;
    bool remove_preprocessed;
    bool shuffle_input_edges;
    torch::Dtype edges_dtype;
    string s_edges_dtype;
    BackendType embeddings_backend_type;
    string s_embeddings_backend_type;
    bool reinit_embeddings;
    BackendType relations_backend_type;
    string s_relations_backend_type;
    torch::Dtype embeddings_dtype;
    string s_embeddings_dtype;
    EdgeBucketOrdering edge_bucket_ordering;
    string s_edge_bucket_ordering;
    int num_partitions;
    int buffer_capacity;
    bool prefetching;
    bool conserve_memory;

    // Training options
    int training_batch_size;
    int training_num_chunks;
    int training_negatives;
    float training_degree_fraction;
    NegativeSamplingAccess training_negative_sampling_access;
    string s_training_negative_sampling_access;
    float learning_rate;
    float regularization_coef;
    int regularization_norm;
    OptimizerType optimizer_type;
    string s_optimizer_type;
    LossFunctionType loss_function_type;
    string s_loss_function_type;
    float margin;
    bool average_gradients;
    bool synchronous;
    int num_epochs;
    int checkpoint_interval;
    int shuffle_interval;

    // Training pipeline options
    int max_batches_in_flight;
    bool update_in_flight;
    int embeddings_host_queue_size;
    int embeddings_device_queue_size;
    int gradients_host_queue_size;
    int gradients_device_queue_size;
    int num_embedding_loader_threads;
    int num_embedding_transfer_threads;
    int num_compute_threads;
    int num_gradient_transfer_threads;
    int num_embedding_update_threads;

    // Evaluation options
    int evaluation_batch_size;
    int evaluation_num_chunks;
    int evaluation_negatives;
    float evaluation_degree_fraction;
    NegativeSamplingAccess eval_negative_sampling_access;
    string s_eval_negative_sampling_access;
    int epochs_per_eval;
    bool eval_synchronous;
    string s_evaluation_method;
    bool filtered_eval;
    int checkpoint_to_eval;

    // Evaluation pipeline options
    int evaluate_max_batches_in_flight;
    int evaluate_embeddings_host_queue_size;
    int evaluate_embeddings_device_queue_size;
    int evaluate_num_embedding_loader_threads;
    int evaluate_num_embedding_transfer_threads;
    int num_evaluate_threads;

    // Path options
    string train_edges;
    string train_edges_partitions;
    string validation_edges;
    string validation_edges_partitions;
    string test_edges;
    string test_edges_partitions;
    string node_labels;
    string relation_labels;
    string node_ids;
    string relation_ids;
    string custom_ordering;
    string base_directory;

    // Reporting options
    int logs_per_epoch;
    spdlog::level::level_enum log_level;
    string s_log_level;

    INIReader reader(config_path);

    if (reader.ParseError() != 0)
        SPDLOG_ERROR("Can't load {}", config_path);

    // General
    s_device = reader.Get("general", "device", "CPU"); // Device to use for training
    s_gpu_ids = reader.Get("general", "gpu_ids", "0"); // Ids of the gpus to use
    rand_seed = reader.GetInteger("general", "random_seed", time(0)); // Random seed to use
    num_train = reader.GetInteger("general", "num_train", -1); // Number of edges in the graph
    num_valid = reader.GetInteger("general", "num_valid", 0); // Number of edges in the graph
    num_test = reader.GetInteger("general", "num_test", 0); // Number of edges in the graph
    num_nodes = reader.GetInteger("general", "num_nodes", -1); // Number of nodes in the graph
    num_relations = reader.GetInteger("general", "num_relations", -1); // Number of relations in the graph
    experiment_name = reader.Get("general", "experiment_name", "marius"); // Name for the current experiment

    // Model
    scale_factor = reader.GetFloat("model", "scale_factor", .001); // Factor to scale the embeddings upon initialization
    s_initialization_distribution = reader.Get("model", "initialization_distribution", "Normal"); // Which distribution to use for initializing embeddings
    embedding_size = reader.GetInteger("model", "embedding_size", 128); // Dimension of the embedding vectors
    s_encoder_model = reader.Get("model", "encoder", "None"); // Encoder model to use
    s_decoder_model = reader.Get("model", "decoder", "DistMult"); // Decoder model to use

    // Storage
    s_edges_backend_type = reader.Get("storage", "edges_backend", "HostMemory"); // Storage backend to use
    reinit_edges = reader.GetBoolean("storage", "reinit_edges", true); // If true, the edges in the data directory will be reinitialized
    remove_preprocessed = reader.GetBoolean("storage", "remove_preprocessed", false); // If true, the input edge files will be removed
    shuffle_input_edges = reader.GetBoolean("storage", "shuffle_input_edges", true); // If true, the input edge files will be shuffled
    s_edges_dtype = reader.Get("storage", "edges_dtype", "int32"); // Type of the embedding vectors
    s_embeddings_backend_type = reader.Get("storage", "embeddings_backend", "HostMemory"); // Storage backend to use
    reinit_embeddings = reader.GetBoolean("storage", "reinit_embeddings", true); // If true, the embeddings in the data directory will be reinitialized
    s_relations_backend_type = reader.Get("storage", "relations_backend", "HostMemory"); // Storage backend to use
    s_embeddings_dtype = reader.Get("storage", "embeddings_dtype", "float32"); // Type of the embedding vectors
    s_edge_bucket_ordering = reader.Get("storage", "edge_bucket_ordering", "Elimination"); // How to order edge buckets
    num_partitions = reader.GetInteger("storage", "num_partitions", 1); // Number of partitions for training
    buffer_capacity = reader.GetInteger("storage", "buffer_capacity", 2); // Number of partitions to hold in memory
    prefetching = reader.GetBoolean("storage", "prefetching", true); // Whether to prefetch partitions or not
    conserve_memory = reader.GetBoolean("storage", "conserve_memory", false); // Will try to conserve memory at the cost of extra IO for some configurations

    // Training
    training_batch_size = reader.GetInteger("training", "batch_size", 10000); // Number of positive edges in a batch
    training_num_chunks = reader.GetInteger("training", "number_of_chunks", 16); // Number of chunks to split up positives into
    training_negatives = reader.GetInteger("training", "negatives", 512); // Number of negatives to sample per chunk
    training_degree_fraction = reader.GetFloat("training", "degree_fraction", .5); // Fraction of negatives which are sampled by degree
    s_training_negative_sampling_access = reader.Get("training", "negative_sampling_access", "Uniform"); // How negative samples are generated
    learning_rate = reader.GetFloat("training", "learning_rate", .1); // Learning rate to use
    regularization_coef = reader.GetFloat("training", "regularization_coef", 2e-6); // Regularization Coefficient
    regularization_norm = reader.GetInteger("training", "regularization_norm", 2); // Norm of the regularization
    s_optimizer_type = reader.Get("training", "optimizer", "Adagrad"); // Optimizer to use
    s_loss_function_type = reader.Get("training", "loss", "SoftMax"); // Loss to use
    margin = reader.GetFloat("training", "margin", 0); // Margin to use in ranking loss
    average_gradients = reader.GetBoolean("training", "average_gradients", false); // If true gradients will be averaged when accumulated, summed if false
    synchronous = reader.GetBoolean("training", "synchronous", false); // If true training will be synchronous
    num_epochs = reader.GetInteger("training", "num_epochs", 10); // Number of epochs to train
    checkpoint_interval = reader.GetInteger("training", "checkpoint_interval", 9999); // Will checkpoint model after each interval of epochs
    shuffle_interval = reader.GetInteger("training", "shuffle_interval", 1); // How many epochs until a shuffle of the edges is performed

    // Training Pipeline
    max_batches_in_flight = reader.GetInteger("training_pipeline", "max_batches_in_flight", 16); // Vary the amount of batches allowed in the pipeline at once
    update_in_flight = reader.GetBoolean("training_pipeline", "update_in_flight", false); // If true, batches in the pipeline will receive gradient updates
    embeddings_host_queue_size = reader.GetInteger("training_pipeline", "embeddings_host_queue_size", 4); // Size of embeddings host queue
    embeddings_device_queue_size = reader.GetInteger("training_pipeline", "embeddings_device_queue_size", 4); // Size of embeddings device queue
    gradients_host_queue_size = reader.GetInteger("training_pipeline", "gradients_host_queue_size", 4); // Size of gradients host queue
    gradients_device_queue_size = reader.GetInteger("training_pipeline", "gradients_device_queue_size", 4); // Size of gradients device queue
    num_embedding_loader_threads = reader.GetInteger("training_pipeline", "num_embedding_loader_threads", 2); // Number of embedding loader threads
    num_embedding_transfer_threads = reader.GetInteger("training_pipeline", "num_embedding_transfer_threads", 2); // Number of embedding transfer threads
    num_compute_threads = reader.GetInteger("training_pipeline", "num_compute_threads", 1); // Number of compute threads
    num_gradient_transfer_threads = reader.GetInteger("training_pipeline", "num_gradient_transfer_threads", 2); // Number of gradient transfer threads
    num_embedding_update_threads = reader.GetInteger("training_pipeline", "num_embedding_update_threads", 2); // Number of embedding updater threads

    // Evaluation
    evaluation_batch_size = reader.GetInteger("evaluation", "batch_size", 1000); // Number of positive edges in a batch
    evaluation_num_chunks = reader.GetInteger("evaluation", "number_of_chunks", 1); // Number of chunks to split up positives into
    evaluation_negatives = reader.GetInteger("evaluation", "negatives", 1000); // Number of negatives to sample per chunk
    evaluation_degree_fraction = reader.GetFloat("evaluation", "degree_fraction", .5); // Fraction of negatives to sample by degree
    s_eval_negative_sampling_access = reader.Get("evaluation", "negative_sampling_access", "Uniform"); // Negative sampling policy to use for evaluation
    epochs_per_eval = reader.GetInteger("evaluation", "epochs_per_eval", 1); // Number of positive edges in a batch
    eval_synchronous = reader.GetBoolean("evaluation", "synchronous", false); // Amount of data to hold out for validation set
    s_evaluation_method = reader.Get("evaluation", "evaluation_method", "LinkPrediction"); // Evaluation method to use
    filtered_eval = reader.GetBoolean("evaluation", "filtered_evaluation", false); // If true false negatives will be filtered
    checkpoint_to_eval = reader.GetInteger("evaluation", "checkpoint_id", -1); // Checkpoint to evaluate

    // Evalutaion Pipeline
    evaluate_max_batches_in_flight = reader.GetInteger("evaluation_pipeline", "max_batches_in_flight", 32); // Vary the amount of batches allowed in the pipeline at once
    evaluate_embeddings_host_queue_size = reader.GetInteger("evaluation_pipeline", "embeddings_host_queue_size", 8); // Size of embeddings host queue
    evaluate_embeddings_device_queue_size = reader.GetInteger("evaluation_pipeline", "embeddings_device_queue_size", 8); // Size of embeddings device queue
    evaluate_num_embedding_loader_threads = reader.GetInteger("evaluation_pipeline", "num_embedding_loader_threads", 4); // Number of embedding loader threads
    evaluate_num_embedding_transfer_threads = reader.GetInteger("evaluation_pipeline", "num_embedding_transfer_threads", 4); // Number of embedding transfer threads
    num_evaluate_threads = reader.GetInteger("evaluation_pipeline", "num_evaluate_threads", 1); // Number of evaluate threads

    // Path
    train_edges = reader.Get("path", "train_edges", ""); // Path to training edges file
    if (train_edges == "")
        SPDLOG_ERROR("Path to training edges required");
    train_edges_partitions = reader.Get("path", "train_edges_partitions", ""); // Path to training edge partition file
    validation_edges = reader.Get("path", "validation_edges", ""); // Path to validation edges file
    validation_edges_partitions = reader.Get("path", "validation_partitions", ""); // Path to training edge partition file
    test_edges = reader.Get("path", "test_edges", ""); // Path to edges used for testing
    test_edges_partitions = reader.Get("path", "test_edges_partitions", ""); // Path to training edge partition file
    node_labels = reader.Get("path", "node_labels", ""); // Path to node labels for Node classification
    relation_labels = reader.Get("path", "relation_labels", ""); // Path to relation labels for Relation classification
    node_ids = reader.Get("path", "node_ids", ""); // Path to node ids
    relation_ids = reader.Get("path", "relations_ids", ""); // Path to relations ids
    custom_ordering = reader.Get("path", "custom_ordering", ""); // Path to file where edge bucket ordering is stored
    base_directory = reader.Get("path", "base_directory", "data/"); // Path to directory where data is stored

    // Reporting
    logs_per_epoch = reader.GetInteger("reporting", "logs_per_epoch", 10); // How many times log statements will be output during a single epoch of training or evaluation
    s_log_level = reader.Get("reporting", "log_level", "info"); // Log level to use

//    po::options_description config_options("Configuration");
//    po::variables_map variables_map;
//
//    try {
//        store(parse_command_line(argc, argv, config_options), variables_map);
//        store(parse_config_file(config_fstream, config_options), variables_map);
//        notify(variables_map);
//    } catch(std::exception& e) {
//        SPDLOG_ERROR(e.what());
//        exit(-1);
//    }

    if (s_device == "GPU") {
        device = torch::kCUDA;
    } else if (s_device == "CPU") {
        device = torch::kCPU;
    } else {
        SPDLOG_ERROR("Unrecognized device type {}. Options are [GPU, CPU].", s_device);
        exit(-1);
    }

    std::stringstream ss(s_gpu_ids);
    int number;
    while (ss >> number)
        gpu_ids.push_back(number);

    if (s_initialization_distribution == "Uniform") {
        initialization_distribution = InitializationDistribution::Uniform;
    } else if (s_initialization_distribution == "Normal") {
        initialization_distribution = InitializationDistribution::Normal;
    } else {
        SPDLOG_ERROR("Unrecognized Distribution: {}. Options are [Uniform, Normal]", s_initialization_distribution);
        exit(-1);
    }

    if (s_encoder_model == "None") {
        encoder_model = EncoderModelType::None;
    } else if (s_encoder_model == "Custom") {
        encoder_model = EncoderModelType::None;
    } else {
        SPDLOG_ERROR("Unrecognized Encoder Model: {}. Options are [None, Custom]", s_encoder_model);
        exit(-1);
    }

    RelationOperatorType relation_operator;
    ComparatorType comparator;
    if (s_decoder_model == "NodeClassification") {
        decoder_model = DecoderModelType::NodeClassification;
    } else if (s_decoder_model == "DistMult") {
        decoder_model = DecoderModelType::DistMult;
        comparator = ComparatorType::Dot;
        relation_operator = RelationOperatorType::Hadamard;
    } else if (s_decoder_model == "TransE") {
        decoder_model = DecoderModelType::TransE;
        comparator = ComparatorType::Cosine;
        relation_operator = RelationOperatorType::Translation;
    } else if (s_decoder_model == "ComplEx") {
        decoder_model = DecoderModelType::ComplEx;
        comparator = ComparatorType::Dot;
        relation_operator = RelationOperatorType::ComplexHadamard;
    } else {
        SPDLOG_ERROR("Unrecognized Evaluation Method: {}. Options are [NodeClassification, DistMult, TransE, ComplEx]", s_decoder_model);
        exit(-1);
    }

    if (s_edges_backend_type == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // edges_backend_type = BackendType::RocksDB;
    } else if (s_edges_backend_type == "DeviceMemory") {
        edges_backend_type = BackendType::DeviceMemory;
    } else if (s_edges_backend_type == "FlatFile") {
        edges_backend_type = BackendType::FlatFile;
    } else if (s_edges_backend_type == "HostMemory") {
        edges_backend_type = BackendType::HostMemory;
    } else {
        SPDLOG_ERROR("Unrecognized Edge Storage Backend: {}. Options are [DeviceMemory, FlatFile, HostMemory]", s_edges_backend_type);
        exit(-1);
    }

    if (s_edges_dtype == "int32") {
        edges_dtype = torch::kInt32;
    } else if (s_edges_dtype == "int64") {
        edges_dtype = torch::kInt64;
    } else {
        SPDLOG_ERROR("Unrecognized edges datatype {}. Options are [int32, int64].", s_edges_dtype);
        exit(-1);
    }

    if (s_embeddings_backend_type == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // embeddings_backend_type = BackendType::RocksDB;
    } else if (s_embeddings_backend_type == "HostMemory") {
        embeddings_backend_type = BackendType::HostMemory;
    } else if (s_embeddings_backend_type == "DeviceMemory") {
        embeddings_backend_type = BackendType::DeviceMemory;
    } else if (s_embeddings_backend_type == "FlatFile") {
        SPDLOG_ERROR("FlatFile backend unsupported for node embeddings.");
        exit(-1);
        // embeddings_backend_type = BackendType::FlatFile;
    } else if (s_embeddings_backend_type == "PartitionBuffer") {
        embeddings_backend_type = BackendType::PartitionBuffer;
    } else {
        SPDLOG_ERROR("Unrecognized Node Embedding Storage Backend: {}. Options are [DeviceMemory, PartitionBuffer, HostMemory]", s_embeddings_backend_type);
        exit(-1);
    }

    if (s_relations_backend_type == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // embeddings_backend_type = BackendType::RocksDB;
    } else if (s_relations_backend_type == "HostMemory") {
        relations_backend_type = BackendType::HostMemory;
    } else if (s_relations_backend_type == "DeviceMemory") {
        relations_backend_type = BackendType::DeviceMemory;
    } else if (s_relations_backend_type == "FlatFile") {
        SPDLOG_ERROR("FlatFile backend unsupported for relation embeddings.");
        exit(-1);
        // embeddings_backend_type = BackendType::FlatFile;
    } else if (s_relations_backend_type == "PartitionBuffer") {
        SPDLOG_ERROR("PartitionBuffer backend unsupported for relation embeddings.");
        exit(-1);
    } else {
        SPDLOG_ERROR("Unrecognized Relation Embedding Storage Backend: {}. Options are [DeviceMemory, HostMemory]", s_relations_backend_type);
        exit(-1);
    }

    if (s_embeddings_dtype == "float16") {
        embeddings_dtype = torch::kFloat16;
    } else if (s_embeddings_dtype == "float32") {
        embeddings_dtype = torch::kFloat32;
    } else if (s_embeddings_dtype == "float64") {
        embeddings_dtype = torch::kFloat64;
    } else {
        SPDLOG_ERROR("Unrecognized embedding datatype {}. Options are [float16, float32, float64].", s_embeddings_dtype);
        exit(-1);
    }

    if (s_edge_bucket_ordering == "Hilbert") {
        edge_bucket_ordering = EdgeBucketOrdering::Hilbert;
    } else if (s_edge_bucket_ordering == "HilbertSymmetric") {
        edge_bucket_ordering = EdgeBucketOrdering::HilbertSymmetric;
    } else if (s_edge_bucket_ordering == "Random") {
        edge_bucket_ordering = EdgeBucketOrdering::Random;
    } else if (s_edge_bucket_ordering == "RandomSymmetric") {
        edge_bucket_ordering = EdgeBucketOrdering::RandomSymmetric;
    } else if (s_edge_bucket_ordering == "Sequential") {
        edge_bucket_ordering = EdgeBucketOrdering::Sequential;
    } else if (s_edge_bucket_ordering == "SequentialSymmetric") {
        edge_bucket_ordering = EdgeBucketOrdering::SequentialSymmetric;
    } else if (s_edge_bucket_ordering == "Elimination") {
        edge_bucket_ordering = EdgeBucketOrdering::Elimination;
    } else if (s_edge_bucket_ordering == "Custom") {
        edge_bucket_ordering = EdgeBucketOrdering::Custom;
        // assert custom ordering file exists
    } else {
        SPDLOG_ERROR("Unrecognized Edge Bucket Ordering: {}. Options are [Hilbert, HilbertSymmetric, Random, RandomSymmetric, Sequential, SequentialSymmetric, Elimination, Custom]", s_edge_bucket_ordering);
        exit(-1);
    }

    if (s_training_negative_sampling_access == "UniformCrossPartition") {
        training_negative_sampling_access = NegativeSamplingAccess::UniformCrossPartition;
    } else if (s_training_negative_sampling_access == "Uniform" ){
        training_negative_sampling_access = NegativeSamplingAccess::Uniform;
    } else {
        SPDLOG_ERROR("Unrecognized Negative Sampling Access Policy: {}. Options are [UniformCrossPartition, Uniform]", s_training_negative_sampling_access);
        exit(-1);
    }

    if (s_optimizer_type == "SGD") {
        SPDLOG_ERROR("SGD Currently Unsupported");
        exit(-1);
//        optimizer_type = OptimizerType::SGD;
    } else if (s_optimizer_type == "Adagrad") {
        optimizer_type = OptimizerType::Adagrad;
    } else {
        SPDLOG_ERROR("Unrecognized optimizer type {}. Options are [SGD, Adagrad].", s_optimizer_type);
        exit(-1);
    }

    if (s_loss_function_type == "Ranking") {
        loss_function_type = LossFunctionType::RankingLoss;
    } else if (s_loss_function_type == "SoftMax") {
        loss_function_type = LossFunctionType::SoftMax;
    } else {
        SPDLOG_ERROR("Unrecognized loss function {}. Options are [Ranking, SoftMax].", s_loss_function_type);
        exit(-1);
    }
    
    if (s_eval_negative_sampling_access == "UniformCrossPartition") {
        eval_negative_sampling_access = NegativeSamplingAccess::UniformCrossPartition;
    } else if (s_eval_negative_sampling_access == "Uniform" ){
        eval_negative_sampling_access = NegativeSamplingAccess::Uniform;
    } else if (s_eval_negative_sampling_access == "All" ){
        eval_negative_sampling_access = NegativeSamplingAccess::All;
    } else {
        SPDLOG_ERROR("Unrecognized Negative Sampling Access Policy: {}. Options are [UniformCrossPartition, Uniform, All]", s_eval_negative_sampling_access);
        exit(-1);
    }

    if (s_log_level == "info") {
        log_level = spdlog::level::info;
    } else if (s_log_level == "debug") {
        log_level = spdlog::level::debug;
    } else if (s_log_level == "trace") {
        log_level = spdlog::level::trace;
    } else {
        SPDLOG_ERROR("Unrecognized log level: {}. Options are [info, debug, trace]", s_log_level);
        exit(-1);
    }

    GeneralOptions general_options = {
        device,
        gpu_ids,
        rand_seed,
        num_train,
        num_valid,
        num_test,
        num_nodes,
        num_relations,
        experiment_name
    };
    
    ModelOptions model_options = {
        scale_factor,
        initialization_distribution,
        embedding_size,
        encoder_model,
        decoder_model,
        comparator,
        relation_operator
    };

    StorageOptions storage_options = {
        edges_backend_type,
        reinit_edges,
        remove_preprocessed,
        shuffle_input_edges,
        edges_dtype,
        embeddings_backend_type,
        reinit_embeddings,
        relations_backend_type,
        embeddings_dtype,
        edge_bucket_ordering,
        num_partitions,
        buffer_capacity,
        prefetching,
        conserve_memory
    };

    TrainingOptions training_options = {
        training_batch_size,
        training_num_chunks,
        training_negatives,
        training_degree_fraction,
        training_negative_sampling_access,
        learning_rate,
        regularization_coef,
        regularization_norm,
        optimizer_type,
        loss_function_type,
        margin,
        average_gradients,
        synchronous,
        num_epochs,
        checkpoint_interval,
        shuffle_interval
    };

    TrainingPipelineOptions training_pipeline_options = {
        max_batches_in_flight,
        update_in_flight,
        embeddings_host_queue_size,
        embeddings_device_queue_size,
        gradients_host_queue_size,
        gradients_device_queue_size,
        num_embedding_loader_threads,
        num_embedding_transfer_threads,
        num_compute_threads,
        num_gradient_transfer_threads,
        num_embedding_update_threads
    };

    EvaluationOptions evaluation_options = {
        evaluation_batch_size,
        evaluation_num_chunks,
        evaluation_negatives,
        evaluation_degree_fraction,
        eval_negative_sampling_access,
        epochs_per_eval,
        eval_synchronous,
        filtered_eval,
        checkpoint_to_eval
    };

    EvaluationPipelineOptions evaluation_pipeline_options = {
        evaluate_max_batches_in_flight,
        evaluate_embeddings_host_queue_size,
        evaluate_embeddings_device_queue_size,
        evaluate_num_embedding_loader_threads,
        evaluate_num_embedding_transfer_threads,
        num_evaluate_threads
    };

    PathOptions path_options = {
        train_edges,
        train_edges_partitions,
        validation_edges,
        validation_edges_partitions,
        test_edges,
        test_edges_partitions,
        node_labels,
        relation_labels,
        node_ids,
        relation_ids,
        custom_ordering,
        base_directory,
        base_directory + experiment_name + "/"
    };

    ReportingOptions reporting_options {
        logs_per_epoch,
        log_level
    };

    MariusOptions options = {
        general_options,
        model_options,
        storage_options,
        training_options,
        training_pipeline_options,
        evaluation_options,
        evaluation_pipeline_options,
        path_options,
        reporting_options
    };

    if (validateNumericalOptions(options) == false) {
        exit(-1);
    }

    return options;
}

bool validateNumericalOptions(MariusOptions options) {

    struct IntValueRange {
        string name;
        int64_t value;
        int64_t range[2];
    };

    struct FloatValueRange {
        string name;
        float value;
        float range[2];
    };

    std::vector<IntValueRange> intRanges;
    std::vector<FloatValueRange> floatRanges;

    float float_max = std::numeric_limits<float>::max();

    // GENERAL OPTIONS
    IntValueRange rand_seed = {"general.random_seed", options.general.random_seed, {0, INT64_MAX}};
    intRanges.push_back(rand_seed);
    IntValueRange num_train = {"general.num_train", options.general.num_train, {0, INT64_MAX}};
    intRanges.push_back(num_train);
    IntValueRange num_valid = {"general.num_valid", options.general.num_valid, {0, INT64_MAX}};
    intRanges.push_back(num_valid);
    IntValueRange num_test = {"general.num_test", options.general.num_test, {0, INT64_MAX}};
    intRanges.push_back(num_test);
    IntValueRange num_nodes = {"general.num_nodes", options.general.num_nodes, {0, INT64_MAX}};
    intRanges.push_back(num_nodes);
    IntValueRange num_relations = {"general.num_relations", options.general.num_relations, {0, INT64_MAX}};
    intRanges.push_back(num_relations);

    // MODEL OPTIONS
    FloatValueRange scale_factor = {"model.scale_factor", options.model.scale_factor, {0, float_max}};
    floatRanges.push_back(scale_factor);
    IntValueRange embedding_size = {"model.embedding_size", options.model.embedding_size, {1, INT32_MAX}};
    intRanges.push_back(embedding_size);

    // STORAGE OPTIONS
    IntValueRange num_partitions = {"storage.num_partitions", options.storage.num_partitions, {1, INT32_MAX}};
    intRanges.push_back(num_partitions);
    IntValueRange buffer_capacity = {"storage.buffer_capacity", options.storage.buffer_capacity, {2, INT32_MAX}};
    intRanges.push_back(buffer_capacity);

    // TRAINING OPTIONS
    IntValueRange training_batch_size = {"training.batch_size", options.training.batch_size, {1, INT32_MAX}};
    intRanges.push_back(training_batch_size);
    IntValueRange training_num_chunks = {"training.number_of_chunks", options.training.number_of_chunks, {1, options.training.batch_size}};
    intRanges.push_back(training_num_chunks);
    IntValueRange training_negatives = {"training.negatives", options.training.negatives, {1, INT32_MAX}};
    intRanges.push_back(training_negatives);
    FloatValueRange training_degree_fraction = {"training.degree_fraction", options.training.degree_fraction, {0.0, 1.0}};
    floatRanges.push_back(training_degree_fraction);
    FloatValueRange learning_rate = {"training.learning_rate", options.training.learning_rate, {0, float_max}};
    floatRanges.push_back(learning_rate);
    FloatValueRange regularization_coef = {"training.regularization_coef", options.training.regularization_coef, {0, float_max}};
    floatRanges.push_back(regularization_coef);
    IntValueRange regularization_norm = {"training.regularization_norm", options.training.regularization_norm, {0, INT32_MAX}};
    intRanges.push_back(regularization_norm);
    FloatValueRange margin = {"training.margin", options.training.margin, {0, float_max}};
    floatRanges.push_back(margin);
    IntValueRange num_epochs = {"training.num_epochs", options.training.num_epochs, {0, INT32_MAX}};
    intRanges.push_back(num_epochs);
    IntValueRange checkpoint_interval = {"training.checkpoint_interval", options.training.num_epochs, {1, INT32_MAX}};
    intRanges.push_back(checkpoint_interval);
    IntValueRange shuffle_interval = {"training.shuffle_interval", options.training.shuffle_interval, {1, INT32_MAX}};
    intRanges.push_back(shuffle_interval);

    // TRAINING PIPELINE OPTIONS
    IntValueRange max_batches_in_flight = {"training_pipeline.max_batches_in_flight", options.training_pipeline.max_batches_in_flight, {1, INT32_MAX}};
    intRanges.push_back(max_batches_in_flight);
    IntValueRange embeddings_host_queue_size = {"training_pipeline.embeddings_host_queue_size", options.training_pipeline.embeddings_host_queue_size, {1, INT32_MAX}};
    intRanges.push_back(embeddings_host_queue_size);
    IntValueRange embeddings_device_queue_size = {"training_pipeline.embeddings_device_queue_size", options.training_pipeline.embeddings_device_queue_size, {1, INT32_MAX}};
    intRanges.push_back(embeddings_device_queue_size);
    IntValueRange gradients_host_queue_size = {"training_pipeline.gradients_host_queue_size", options.training_pipeline.gradients_host_queue_size, {1, INT32_MAX}};
    intRanges.push_back(gradients_host_queue_size);
    IntValueRange gradients_device_queue_size = {"training_pipeline.gradients_device_queue_size", options.training_pipeline.gradients_device_queue_size, {1, INT32_MAX}};
    intRanges.push_back(gradients_device_queue_size);
    IntValueRange num_embedding_loader_threads = {"training_pipeline.num_embedding_loader_threads", options.training_pipeline.num_embedding_loader_threads, {1, INT32_MAX}};
    intRanges.push_back(num_embedding_loader_threads);
    IntValueRange num_embedding_transfer_threads = {"training_pipeline.num_embedding_transfer_threads", options.training_pipeline.num_embedding_transfer_threads, {1, INT32_MAX}};
    intRanges.push_back(num_embedding_transfer_threads);
    IntValueRange num_compute_threads = {"training_pipeline.num_compute_threads", options.training_pipeline.num_compute_threads, {1, INT32_MAX}};
    intRanges.push_back(num_compute_threads);
    IntValueRange num_gradient_transfer_threads = {"training_pipeline.num_gradient_transfer_threads", options.training_pipeline.num_gradient_transfer_threads, {1, INT32_MAX}};
    intRanges.push_back(num_gradient_transfer_threads);
    IntValueRange num_embedding_update_threads = {"training_pipeline.num_embedding_update_threads", options.training_pipeline.num_embedding_update_threads, {1, INT32_MAX}};
    intRanges.push_back(num_embedding_update_threads);

    // EVALUATION OPTIONS
    IntValueRange evaluation_batch_size = {"evaluation.batch_size", options.evaluation.batch_size, {1, INT32_MAX}};
    intRanges.push_back(evaluation_batch_size);
    IntValueRange evaluation_num_chunks = {"evaluation.number_of_chunks", options.evaluation.number_of_chunks, {0, INT64_MAX}};
    intRanges.push_back(evaluation_num_chunks);
    IntValueRange evaluation_negatives = {"evaluation.negatives", options.evaluation.negatives, {0, INT64_MAX}};
    intRanges.push_back(evaluation_negatives);
    FloatValueRange evaluation_degree_fraction = {"evaluation.degree_fraction", options.evaluation.degree_fraction, {0.0, 1.0}};
    floatRanges.push_back(evaluation_degree_fraction);
    IntValueRange epochs_per_eval = {"evaluation.epochs_per_eval", options.evaluation.epochs_per_eval, {1, INT32_MAX}};
    intRanges.push_back(epochs_per_eval);
    IntValueRange checkpoint_to_eval = {"evaluation.checkpoint_to_eval", options.evaluation.checkpoint_to_eval, {-1, INT32_MAX}};
    intRanges.push_back(checkpoint_to_eval);

    // EVALUATION PIPELINE OPTIONS
    IntValueRange evaluate_max_batches_in_flight = {"evaluation_pipeline.max_batches_in_flight", options.evaluation_pipeline.max_batches_in_flight, {1, INT32_MAX}};
    intRanges.push_back(evaluate_max_batches_in_flight);
    IntValueRange evaluate_embeddings_host_queue_size = {"evaluation_pipeline.embeddings_host_queue_size", options.evaluation_pipeline.embeddings_host_queue_size, {1, INT32_MAX}};
    intRanges.push_back(evaluate_embeddings_host_queue_size);
    IntValueRange evaluate_embeddings_device_queue_size = {"evaluation_pipeline.embeddings_device_queue_size", options.evaluation_pipeline.embeddings_device_queue_size, {1, INT32_MAX}};
    intRanges.push_back(evaluate_embeddings_device_queue_size);
    IntValueRange evaluate_num_embedding_loader_threads = {"evaluation_pipeline.num_embedding_loader_threads", options.evaluation_pipeline.num_embedding_loader_threads, {1, INT32_MAX}};
    intRanges.push_back(evaluate_num_embedding_loader_threads);
    IntValueRange evaluate_num_embedding_transfer_threads = {"evaluation_pipeline.num_embedding_transfer_threads", options.evaluation_pipeline.num_embedding_transfer_threads, {1, INT32_MAX}};
    intRanges.push_back(evaluate_num_embedding_transfer_threads);
    IntValueRange num_evaluate_threads = {"evaluation_pipeline.num_evaluate_threads", options.evaluation_pipeline.num_evaluate_threads, {0, INT32_MAX}};
    intRanges.push_back(num_evaluate_threads);

    // REPORTING OPTIONS
    IntValueRange logs_per_epoch = {"reporting.logs_per_epoch", options.reporting.logs_per_epoch, {0, INT32_MAX}};
    intRanges.push_back(logs_per_epoch);

    for (IntValueRange v : intRanges) {
        if (v.value < v.range[0] || v.value > v.range[1]) {
            SPDLOG_ERROR("Parameter {}: value {} out of range [{}, {}]", v.name, v.value, v.range[0], v.range[1]);
            return false;
        }
    }
    for (FloatValueRange v : floatRanges) {
        if (v.value < v.range[0] || v.value > v.range[1]) {
            SPDLOG_ERROR("Parameter {}: value {} out of range [{}, {}]", v.name, v.value, v.range[0], v.range[1]);
            return false;
        }
    }

    return true;
}

void logConfig() {
    SPDLOG_DEBUG("########## General Options ##########");

    SPDLOG_DEBUG("########## Storage Options ##########");

    SPDLOG_DEBUG("########## Training Options ##########");

    SPDLOG_DEBUG("########## Training Pipeline Options ##########");

    SPDLOG_DEBUG("########## Evaluation Options ##########");

    SPDLOG_DEBUG("########## Evaluation Pipeline Options ##########");

    SPDLOG_DEBUG("########## Path Options ##########");
}