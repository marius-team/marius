//
// Created by Jason Mohoney on 2/18/20.
//

#include <config.h>

MariusOptions marius_options = MariusOptions();
TimestampAllocator global_timestamp_allocator = TimestampAllocator();

MariusOptions parseConfig(int64_t argc, char *argv[]) {

    string config_path;
    cxxopts::Options cmd_options(argv[0], "Train and evaluate graph embeddings");
    cmd_options.allow_unrecognised_options();
    cmd_options.positional_help("");
    cmd_options.custom_help("config_file [OPTIONS...] [<section>.<option>=<value>...]");
    cmd_options.add_options()
        ("config_file", "Configuration file", cxxopts::value<std::string>())
        ("h, help", "Print help and exit.");

    // Get config
    try {
        cmd_options.parse_positional({"config_file"});
        auto result = cmd_options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << cmd_options.help() << std::endl;
            exit(0);
        }

        if (!result.count("config_file")) {
            std::cout << cmd_options.help() << std::endl;
            throw cxxopts::option_required_exception("config_file");
        }

        config_path = result["config_file"].as<string>();

    } catch (const cxxopts::OptionException& e) {
        SPDLOG_ERROR("Error parsing options: {}", e.what());
        exit(-1);
    }

    std::filesystem::path config_file_path = config_path;
    if (!std::filesystem::exists(config_file_path)){
        SPDLOG_ERROR("Unable to find configuration file: {}", config_path);
        exit(1);
    }

    // Associate each option with:
    // C++ variable, default value, section name, user option name, range
    // List for each type of variable
    std::vector<OptInfo<std::string>> s_var_map;
    std::vector<OptInfo<int64_t>> i64_var_map;
    std::vector<OptInfo<int>> i_var_map;
    std::vector<OptInfo<float>> f_var_map;
    std::vector<OptInfo<bool>> b_var_map;


    float FLOAT_MAX = std::numeric_limits<float>::max();

    // General options
    torch::DeviceType device; // Device to use for training
    string s_device;
    s_var_map.push_back((OptInfo<std::string>){&s_device, "CPU", "general", "device"});
    std::vector<int> gpu_ids; // Ids of the gpus to use
    string s_gpu_ids;
    s_var_map.push_back((OptInfo<std::string>){&s_gpu_ids, "0", "general", "gpu_ids"});
    int64_t random_seed; // Random seed to use
    i64_var_map.push_back((OptInfo<int64_t>){&random_seed, time(0), "general", "random_seed", {0, INT64_MAX}});
    int64_t num_train; // Number of edges in the graph
    i64_var_map.push_back((OptInfo<int64_t>){&num_train, -1, "general", "num_train", {0, INT64_MAX}});
    int64_t num_valid; // Number of edges in the graph
    i64_var_map.push_back((OptInfo<int64_t>){&num_valid, 0, "general", "num_valid", {0, INT64_MAX}});
    int64_t num_test; // Number of edges in the graph
    i64_var_map.push_back((OptInfo<int64_t>){&num_test, 0, "general", "num_test", {0, INT64_MAX}});
    int64_t num_nodes; // Number of nodes in the graph
    i64_var_map.push_back((OptInfo<int64_t>){&num_nodes, -1, "general", "num_nodes", {0, INT64_MAX}});
    int64_t num_relations; // Number of relations in the graph
    i64_var_map.push_back((OptInfo<int64_t>){&num_relations, -1, "general", "num_relations", {0, INT64_MAX}});
    string experiment_name; // Name for the current experiment
    s_var_map.push_back((OptInfo<std::string>){&experiment_name, "marius", "general", "experiment_name"});

    // Model options
    float scale_factor; // Factor to scale the embeddings upon initialization
    f_var_map.push_back((OptInfo<float>){&scale_factor, .001, "model", "scale_factor", {0, FLOAT_MAX}});
    InitializationDistribution initialization_distribution; // Which distribution to use for initializing embeddings
    string s_initialization_distribution;
    s_var_map.push_back((OptInfo<std::string>){&s_initialization_distribution, "Normal", "model", "initialization_distribution"});
    int embedding_size; // Dimension of the embedding vectors
    i_var_map.push_back((OptInfo<int>){&embedding_size, 128, "model", "embedding_size", {1, INT32_MAX}});
    EncoderModelType encoder_model; // Encoder model to use
    string s_encoder;
    s_var_map.push_back((OptInfo<std::string>){&s_encoder, "None", "model", "encoder"});
    DecoderModelType decoder_model; // Decoder model to use
    string s_decoder;
    s_var_map.push_back((OptInfo<std::string>){&s_decoder, "DistMult", "model", "decoder"});

    // Storage options
    BackendType edges_backend_type; // Storage backend to use
    string s_edges_backend;
    s_var_map.push_back((OptInfo<std::string>){&s_edges_backend, "HostMemory", "storage", "edges_backend"});
    bool reinit_edges; // If true, the edges in the data directory will be reinitialized
    b_var_map.push_back((OptInfo<bool>){&reinit_edges, true, "storage", "reinit_edges"});
    bool remove_preprocessed; // If true, the input edge files will be removed
    b_var_map.push_back((OptInfo<bool>){&remove_preprocessed, false, "storage", "remove_preprocessed"});
    bool shuffle_input_edges; // If true, the input edge files will be shuffled
    b_var_map.push_back((OptInfo<bool>){&shuffle_input_edges, true, "storage", "shuffle_input_edges"});
    torch::Dtype edges_dtype; // Type of the embedding vectors
    string s_edges_dtype;
    s_var_map.push_back((OptInfo<std::string>){&s_edges_dtype, "int32", "storage", "edges_dtype"});
    BackendType embeddings_backend_type; // Storage backend to use
    string s_embeddings_backend;
    s_var_map.push_back((OptInfo<std::string>){&s_embeddings_backend, "HostMemory", "storage", "embeddings_backend"});
    bool reinit_embeddings; // If true, the embeddings in the data directory will be reinitialized
    b_var_map.push_back((OptInfo<bool>){&reinit_embeddings, true, "storage", "reinit_embeddings"});
    BackendType relations_backend_type; // Storage backend to use
    string s_relations_backend;
    s_var_map.push_back((OptInfo<std::string>){&s_relations_backend, "HostMemory", "storage", "relations_backend"});
    torch::Dtype embeddings_dtype; // Type of the embedding vectors
    string s_embeddings_dtype;
    s_var_map.push_back((OptInfo<std::string>){&s_embeddings_dtype, "float32", "storage", "embeddings_dtype"});
    EdgeBucketOrdering edge_bucket_ordering; // How to order edge buckets
    string s_edge_bucket_ordering;
    s_var_map.push_back((OptInfo<std::string>){&s_edge_bucket_ordering, "Elimination", "storage", "edge_bucket_ordering"});
    int num_partitions; // Number of partitions for training
    i_var_map.push_back((OptInfo<int>){&num_partitions, 1, "storage", "num_partitions", {1, INT32_MAX}});
    int buffer_capacity; // Number of partitions to hold in memory
    i_var_map.push_back((OptInfo<int>){&buffer_capacity, 2, "storage", "buffer_capacity", {2, INT32_MAX}});
    bool prefetching; // Whether to prefetch partitions or not
    b_var_map.push_back((OptInfo<bool>){&prefetching, true, "storage", "prefetching"});
    bool conserve_memory; // Will try to conserve memory at the cost of extra IO for some configurations
    b_var_map.push_back((OptInfo<bool>){&conserve_memory, false, "storage", "conserve_memory"});

    // Training options
    int training_batch_size; // Number of positive edges in a batch
    i_var_map.push_back((OptInfo<int>){&training_batch_size, 10000, "training", "batch_size", {1, INT32_MAX}});
    int training_num_chunks; // Number of chunks to split up positives into
    i_var_map.push_back((OptInfo<int>){&training_num_chunks, 16, "training", "number_of_chunks", {1, INT32_MAX}});
    int training_negatives; // Number of negatives to sample per chunk
    i_var_map.push_back((OptInfo<int>){&training_negatives, 512, "training", "negatives", {1, INT32_MAX}});
    float training_degree_fraction; // Fraction of negatives which are sampled by degree
    f_var_map.push_back((OptInfo<float>){&training_degree_fraction, .5, "training", "degree_fraction", {0, 1.0}});
    NegativeSamplingAccess training_negative_sampling_access; // How negative samples are generated
    string s_training_negative_sampling_access;
    s_var_map.push_back((OptInfo<std::string>){&s_training_negative_sampling_access, "Uniform", "training", "negative_sampling_access"});
    float learning_rate; // Learning rate to use
    f_var_map.push_back((OptInfo<float>){&learning_rate, .1, "training", "learning_rate", {0, FLOAT_MAX}});
    float regularization_coef; // Regularization Coefficient
    f_var_map.push_back((OptInfo<float>){&regularization_coef, 2e-6, "training", "regularization_coef", {0, FLOAT_MAX}});
    int regularization_norm; // Norm of the regularization
    i_var_map.push_back((OptInfo<int>){&regularization_norm, 2, "training", "regularization_norm", {0, INT32_MAX}});
    OptimizerType optimizer_type; // Optimizer to use
    string s_optimizer_type; s_var_map.push_back((OptInfo<std::string>){&s_optimizer_type, "Adagrad", "training", "optimizer"});
    LossFunctionType loss_function_type; // Loss to use
    string s_loss_function_type;
    s_var_map.push_back((OptInfo<std::string>){&s_loss_function_type, "SoftMax", "training", "loss"});
    float margin; // Margin to use in ranking loss
    f_var_map.push_back((OptInfo<float>){&margin, 0, "training", "margin", {0, FLOAT_MAX}});
    bool average_gradients; // If true gradients will be averaged when accumulated, summed if false
    b_var_map.push_back((OptInfo<bool>){&average_gradients, false, "training", "average_gradients"});
    bool synchronous; // If true training will be synchronous
    b_var_map.push_back((OptInfo<bool>){&synchronous, false, "training", "synchronous"});
    int num_epochs; // Number of epochs to train
    i_var_map.push_back((OptInfo<int>){&num_epochs, 10, "training", "num_epochs", {1, INT32_MAX}});
    int checkpoint_interval; // Will checkpoint model after each interval of epochs
    i_var_map.push_back((OptInfo<int>){&checkpoint_interval, 9999, "training", "checkpoint_interval", {1, INT32_MAX}});
    int shuffle_interval; // How many epochs until a shuffle of the edges is performed
    i_var_map.push_back((OptInfo<int>){&shuffle_interval, 1, "training", "shuffle_interval", {1, INT32_MAX}});

    // Training pipeline options
    int max_batches_in_flight; // Vary the amount of batches allowed in the pipeline at once
    i_var_map.push_back((OptInfo<int>){&max_batches_in_flight, 16, "training_pipeline", "max_batches_in_flight", {1, INT32_MAX}});
    bool update_in_flight; // If true, batches in the pipeline will receive gradient updates
    b_var_map.push_back((OptInfo<bool>){&update_in_flight, false, "training_pipeline", "update_in_flight"});
    int embeddings_host_queue_size; // Size of embeddings host queue
    i_var_map.push_back((OptInfo<int>){&embeddings_host_queue_size, 4, "training_pipeline", "embeddings_host_queue_size", {1, INT32_MAX}});
    int embeddings_device_queue_size; // Size of embeddings device queue
    i_var_map.push_back((OptInfo<int>){&embeddings_device_queue_size, 4, "training_pipeline", "embeddings_device_queue_size", {1, INT32_MAX}});
    int gradients_host_queue_size; // Size of gradients host queue
    i_var_map.push_back((OptInfo<int>){&gradients_host_queue_size, 4, "training_pipeline", "gradients_host_queue_size", {1, INT32_MAX}});
    int gradients_device_queue_size; // Size of gradients device queue
    i_var_map.push_back((OptInfo<int>){&gradients_device_queue_size, 4, "training_pipeline", "gradients_device_queue_size", {1, INT32_MAX}});
    int num_embedding_loader_threads; // Number of embedding loader threads
    i_var_map.push_back((OptInfo<int>){&num_embedding_loader_threads, 2, "training_pipeline", "num_embedding_loader_threads", {1, INT32_MAX}});
    int num_embedding_transfer_threads; // Number of embedding transfer threads
    i_var_map.push_back((OptInfo<int>){&num_embedding_transfer_threads, 2, "training_pipeline", "num_embedding_transfer_threads", {1, INT32_MAX}});
    int num_compute_threads; // Number of compute threads
    i_var_map.push_back((OptInfo<int>){&num_compute_threads, 1, "training_pipeline", "num_compute_threads", {1, INT32_MAX}});
    int num_gradient_transfer_threads; // Number of gradient transfer threads
    i_var_map.push_back((OptInfo<int>){&num_gradient_transfer_threads, 2, "training_pipeline", "num_gradient_transfer_threads", {1, INT32_MAX}});
    int num_embedding_update_threads; // Number of embedding updater threads
    i_var_map.push_back((OptInfo<int>){&num_embedding_update_threads, 2, "training_pipeline", "num_embedding_update_threads", {1, INT32_MAX}});

    // Evaluation options
    int evaluation_batch_size; // Number of positive edges in a batch
    i_var_map.push_back((OptInfo<int>){&evaluation_batch_size, 1000, "evaluation", "batch_size", {1, INT32_MAX}});
    int evaluation_num_chunks; // Number of chunks to split up positives into
    i_var_map.push_back((OptInfo<int>){&evaluation_num_chunks, 1, "evaluation", "number_of_chunks", {0, INT32_MAX}});
    int evaluation_negatives; // Number of negatives to sample per chunk
    i_var_map.push_back((OptInfo<int>){&evaluation_negatives, 1000, "evaluation", "negatives", {0, INT32_MAX}});
    float evaluation_degree_fraction; // Fraction of negatives to sample by degree
    f_var_map.push_back((OptInfo<float>){&evaluation_degree_fraction, .5, "evaluation", "degree_fraction", {0, 1.0}});
    NegativeSamplingAccess eval_negative_sampling_access; // Negative sampling policy to use for evaluation
    string s_eval_negative_sampling_access;
    s_var_map.push_back((OptInfo<std::string>){&s_eval_negative_sampling_access, "Uniform", "evaluation", "negative_sampling_access"});
    int epochs_per_eval; // Number of epochs before evaluation
    i_var_map.push_back((OptInfo<int>){&epochs_per_eval, 1, "evaluation", "epochs_per_eval", {1, INT32_MAX}});
    bool eval_synchronous; // Amount of data to hold out for validation set
    b_var_map.push_back((OptInfo<bool>){&eval_synchronous, false, "evaluation", "synchronous"});
    string s_evaluation_method; // Evaluation method to use
    s_var_map.push_back((OptInfo<std::string>){&s_evaluation_method, "LinkPrediction", "evaluation", "evaluation_method"});
    bool filtered_eval; // If true false negatives will be filtered
    b_var_map.push_back((OptInfo<bool>){&filtered_eval, false, "evaluation", "filtered_evaluation"});
    int checkpoint_to_eval; // Checkpoint to evaluate
    i_var_map.push_back((OptInfo<int>){&checkpoint_to_eval, -1, "evaluation", "checkpoint_id", {-1, INT32_MAX}});

    // Evaluation pipeline options
    int evaluate_max_batches_in_flight; // Vary the amount of batches allowed in the pipeline at once
    i_var_map.push_back((OptInfo<int>){&evaluate_max_batches_in_flight, 32, "evaluation", "max_batches_in_flight", {1, INT32_MAX}});
    int evaluate_embeddings_host_queue_size; // Size of embeddings host queue
    i_var_map.push_back((OptInfo<int>){&evaluate_embeddings_host_queue_size, 8, "evaluation", "embeddings_host_queue_size", {1, INT32_MAX}});
    int evaluate_embeddings_device_queue_size; // Size of embeddings device queue
    i_var_map.push_back((OptInfo<int>){&evaluate_embeddings_device_queue_size, 8, "evaluation", "embeddings_device_queue_size", {1, INT32_MAX}});
    int evaluate_num_embedding_loader_threads; // Number of embedding loader threads
    i_var_map.push_back((OptInfo<int>){&evaluate_num_embedding_loader_threads, 4, "evaluation", "num_embedding_loader_threads", {1, INT32_MAX}});
    int evaluate_num_embedding_transfer_threads; // Number of embedding transfer threads
    i_var_map.push_back((OptInfo<int>){&evaluate_num_embedding_transfer_threads, 4, "evaluation", "num_embedding_transfer_threads", {1, INT32_MAX}});
    int num_evaluate_threads; // Number of evaluate threads
    i_var_map.push_back((OptInfo<int>){&num_evaluate_threads, 1, "evaluation", "num_evaluate_threads", {0, INT32_MAX}});

    // Path options
    string train_edges; // Path to training edges file
    s_var_map.push_back((OptInfo<std::string>){&train_edges, "", "path", "train_edges"});
    string train_edges_partitions; // Path to training edge partition file
    s_var_map.push_back((OptInfo<std::string>){&train_edges_partitions, "", "path", "train_edges_partitions"});
    string validation_edges; // Path to validation edges file
    s_var_map.push_back((OptInfo<std::string>){&validation_edges, "", "path", "validation_edges"});
    string validation_edges_partitions; // Path to validation edge partition file
    s_var_map.push_back((OptInfo<std::string>){&validation_edges_partitions, "", "path", "validation_partitions"});
    string test_edges; // Path to edges used for testing
    s_var_map.push_back((OptInfo<std::string>){&test_edges, "", "path", "test_edges"});
    string test_edges_partitions; // Path to testing edge partition file
    s_var_map.push_back((OptInfo<std::string>){&test_edges_partitions, "", "path", "test_edges_partitions"});
    string node_labels; // Path to node labels for Node classification
    s_var_map.push_back((OptInfo<std::string>){&node_labels, "", "path", "node_labels"});
    string relation_labels; // Path to relation labels for Relation classification
    s_var_map.push_back((OptInfo<std::string>){&relation_labels, "", "path", "relation_labels"});
    string node_ids; // Path to node ids
    s_var_map.push_back((OptInfo<std::string>){&node_ids, "", "path", "node_ids"});
    string relation_ids; // Path to relations ids
    s_var_map.push_back((OptInfo<std::string>){&relation_ids, "", "path", "relations_ids"});
    string custom_ordering; // Path to file where edge bucket ordering is stored
    s_var_map.push_back((OptInfo<std::string>){&custom_ordering, "", "path", "custom_ordering"});
    string base_directory; // Path to directory where data is stored
    s_var_map.push_back((OptInfo<std::string>){&base_directory, "data/", "path", "base_directory"});

    // Reporting options
    int logs_per_epoch; // How many times log statements will be output during a single epoch of training or evaluation
    i_var_map.push_back((OptInfo<int>){&logs_per_epoch, 10, "reporting", "logs_per_epoch", {0, INT32_MAX}});
    spdlog::level::level_enum log_level; // Log level to use
    string s_log_level; 
    s_var_map.push_back((OptInfo<std::string>){&s_log_level, "info", "reporting", "log_level"});

    INIReader reader(config_path);

    if (reader.ParseError() != 0)
        SPDLOG_ERROR("Can't load {}", config_path);

    // Assign values as config or default specifications
    for (OptInfo<std::string> v : s_var_map) {
        *(v.cpp_var) = reader.Get(v.s_section, v.s_option, v.default_val);
    }
    for (OptInfo<int64_t> v : i64_var_map) {
        *(v.cpp_var) = (int64_t)(reader.GetInteger(v.s_section, v.s_option, v.default_val));
    }
    for (OptInfo<int> v : i_var_map) {
        *(v.cpp_var) = reader.GetInteger(v.s_section, v.s_option, v.default_val);
    }
    for (OptInfo<float> v : f_var_map) {
        *(v.cpp_var) = reader.GetFloat(v.s_section, v.s_option, v.default_val);
    }
    for (OptInfo<bool> v : b_var_map) {
        *(v.cpp_var) = reader.GetBoolean(v.s_section, v.s_option, v.default_val);
    }

    // If user specified command line options, overwrite values with these
    try {
        cmd_options.parse_positional({"config_file"});
        auto result = cmd_options.parse(argc, argv);

        if (!result.unmatched().empty()) {
            try {
                for (string opt : result.unmatched()) {
                    if (opt.substr(0, 2) == "--") {

                        int section_end = opt.find(".");
                        int option_end = opt.find("=");

                        if (section_end == - 1 or option_end == - 1) {
                            break;
                        }

                        std::string section = opt.substr(2, section_end - 2);
                        std::string option_name = opt.substr(section_end + 1, option_end - section_end - 1);
                        std::string value = opt.substr(option_end + 1, opt.size() - option_end - 1);

                        for (OptInfo<std::string> v : s_var_map) {
                            if (section == v.s_section && option_name == v.s_option) {
                                *(v.cpp_var) = value;
                            }
                        }
                        for (OptInfo<int64_t> v : i64_var_map) {
                            if (section == v.s_section && option_name == v.s_option) {
                                *(v.cpp_var) = std::stoll(value);
                            }
                        }
                        for (OptInfo<int> v : i_var_map) {
                            if (section == v.s_section && option_name == v.s_option) {
                                *(v.cpp_var) = std::stoi(value);
                            }
                        }
                        for (OptInfo<float> v : f_var_map) {
                            if (section == v.s_section && option_name == v.s_option) {
                                *(v.cpp_var) = std::stof(value);
                            }
                        }
                        for (OptInfo<bool> v : b_var_map) {
                            if (section == v.s_section && option_name == v.s_option) {
                                *(v.cpp_var) = (value == "true");
                            }
                        }
                    } else {
                        throw std::exception();
                    }
                }
            } catch (std::exception ) {
                throw cxxopts::option_syntax_exception("Unable to parse supplied command line configuration options");
            }
        }
    } catch (const cxxopts::OptionException& e) {
        SPDLOG_ERROR("Error parsing options: {}", e.what());
        exit(-1);
    }

    if (train_edges == "")
        SPDLOG_ERROR("Path to training edges required");

    // Validate numerical options
    for (OptInfo<int64_t> v : i64_var_map) {
        if (*(v.cpp_var) < v.range[0] || *(v.cpp_var) > v.range[1]) {
            SPDLOG_ERROR("{}.{}: value {} out of range [{}, {}]", v.s_section, v.s_option, *(v.cpp_var), v.range[0], v.range[1]);
            exit(-1);
        }
    }
    for (OptInfo<int> v : i_var_map) {
        if (*(v.cpp_var) < v.range[0] || *(v.cpp_var) > v.range[1]) {
            SPDLOG_ERROR("{}.{}: value {} out of range [{}, {}]", v.s_section, v.s_option, *(v.cpp_var), v.range[0], v.range[1]);
            exit(-1);
        }
    }
    for (OptInfo<float> v : f_var_map) {
        if (*(v.cpp_var) < v.range[0] || *(v.cpp_var) > v.range[1]) {
            SPDLOG_ERROR("{}.{}: value {} out of range [{}, {}]", v.s_section, v.s_option, *(v.cpp_var), v.range[0], v.range[1]);
            exit(-1);
        }
    }

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

    if (s_encoder == "None") {
        encoder_model = EncoderModelType::None;
    } else if (s_encoder == "Custom") {
        encoder_model = EncoderModelType::None;
    } else {
        SPDLOG_ERROR("Unrecognized Encoder Model: {}. Options are [None, Custom]", s_encoder);
        exit(-1);
    }

    RelationOperatorType relation_operator;
    ComparatorType comparator;
    if (s_decoder == "NodeClassification") {
        decoder_model = DecoderModelType::NodeClassification;
    } else if (s_decoder == "DistMult") {
        decoder_model = DecoderModelType::DistMult;
        comparator = ComparatorType::Dot;
        relation_operator = RelationOperatorType::Hadamard;
    } else if (s_decoder == "TransE") {
        decoder_model = DecoderModelType::TransE;
        comparator = ComparatorType::Cosine;
        relation_operator = RelationOperatorType::Translation;
    } else if (s_decoder == "ComplEx") {
        decoder_model = DecoderModelType::ComplEx;
        comparator = ComparatorType::Dot;
        relation_operator = RelationOperatorType::ComplexHadamard;
    } else {
        SPDLOG_ERROR("Unrecognized Evaluation Method: {}. Options are [NodeClassification, DistMult, TransE, ComplEx]", s_decoder);
        exit(-1);
    }

    if (s_edges_backend == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // edges_backend_type = BackendType::RocksDB;
    } else if (s_edges_backend == "DeviceMemory") {
        edges_backend_type = BackendType::DeviceMemory;
    } else if (s_edges_backend == "FlatFile") {
        edges_backend_type = BackendType::FlatFile;
    } else if (s_edges_backend == "HostMemory") {
        edges_backend_type = BackendType::HostMemory;
    } else {
        SPDLOG_ERROR("Unrecognized Edge Storage Backend: {}. Options are [DeviceMemory, FlatFile, HostMemory]", s_edges_backend);
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

    if (s_embeddings_backend == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // embeddings_backend_type = BackendType::RocksDB;
    } else if (s_embeddings_backend == "HostMemory") {
        embeddings_backend_type = BackendType::HostMemory;
    } else if (s_embeddings_backend == "DeviceMemory") {
        embeddings_backend_type = BackendType::DeviceMemory;
    } else if (s_embeddings_backend == "FlatFile") {
        SPDLOG_ERROR("FlatFile backend unsupported for node embeddings.");
        exit(-1);
        // embeddings_backend_type = BackendType::FlatFile;
    } else if (s_embeddings_backend == "PartitionBuffer") {
        embeddings_backend_type = BackendType::PartitionBuffer;
    } else {
        SPDLOG_ERROR("Unrecognized Node Embedding Storage Backend: {}. Options are [DeviceMemory, PartitionBuffer, HostMemory]", s_embeddings_backend);
        exit(-1);
    }

    if (s_relations_backend == "RocksDB") {
        SPDLOG_ERROR("RocksDB backend currently unsupported.");
        exit(-1);
        // embeddings_backend_type = BackendType::RocksDB;
    } else if (s_relations_backend == "HostMemory") {
        relations_backend_type = BackendType::HostMemory;
    } else if (s_relations_backend == "DeviceMemory") {
        relations_backend_type = BackendType::DeviceMemory;
    } else if (s_relations_backend == "FlatFile") {
        SPDLOG_ERROR("FlatFile backend unsupported for relation embeddings.");
        exit(-1);
        // embeddings_backend_type = BackendType::FlatFile;
    } else if (s_relations_backend == "PartitionBuffer") {
        SPDLOG_ERROR("PartitionBuffer backend unsupported for relation embeddings.");
        exit(-1);
    } else {
        SPDLOG_ERROR("Unrecognized Relation Embedding Storage Backend: {}. Options are [DeviceMemory, HostMemory]", s_relations_backend);
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
        random_seed,
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

    return options;
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