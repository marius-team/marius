
#include <marius.h>

void marius(int argc, char *argv[]) {
    string config_path = argv[1];
    marius_options = parseConfig(config_path, argc, argv);

    std::string log_file = marius_options.general.experiment_name;
    MariusLogger marius_logger = MariusLogger(log_file);
    spdlog::set_default_logger(marius_logger.main_logger_);
    marius_logger.setConsoleLogLevel(marius_options.reporting.log_level);

    bool train = true;
    string path = string(argv[0]);
    string base_filename = path.substr(path.find_last_of("/\\") + 1);
    if (strcmp(base_filename.c_str(), "marius_eval") == 0) {
        train = false;
    }

    if (!train) {
        marius_options.storage.reinitialize_edges = false;
        marius_options.storage.reinitialize_embeddings = false;
    }

    bool gpu = false;
    if (marius_options.general.device == torch::kCUDA) {
        gpu = true;
    }

    Timer preprocessing_timer = Timer(gpu);
    preprocessing_timer.start();
    SPDLOG_INFO("Start preprocessing");

    DataSet *train_set;
    DataSet *eval_set;

    Model *model = initializeModel(marius_options.model.encoder_model, marius_options.model.decoder_model);

    if (train) {
        tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> storage_ptrs = initializeTrain();
        Storage *train_edges = get<0>(storage_ptrs);
        Storage *eval_edges = get<1>(storage_ptrs);
        Storage *test_edges = get<2>(storage_ptrs);

        Storage *embeddings = get<3>(storage_ptrs);
        Storage *emb_state = get<4>(storage_ptrs);

        Storage *src_rel = get<5>(storage_ptrs);
        Storage *src_rel_state = get<6>(storage_ptrs);
        Storage *dst_rel = get<7>(storage_ptrs);
        Storage *dst_rel_state = get<8>(storage_ptrs);

        bool will_evaluate = !(marius_options.path.validation_edges.empty() && marius_options.path.test_edges.empty());

        train_set = new DataSet(train_edges, embeddings, emb_state, src_rel, src_rel_state, dst_rel, dst_rel_state);
        SPDLOG_INFO("Training set initialized");
        if (will_evaluate) {
            eval_set = new DataSet(train_edges, eval_edges, test_edges, embeddings, src_rel, dst_rel);
            SPDLOG_INFO("Evaluation set initialized");
        }

        preprocessing_timer.stop();
        int64_t preprocessing_time = preprocessing_timer.getDuration();

        SPDLOG_INFO("Preprocessing Complete: {}s", (double) preprocessing_time / 1000);

        Trainer *trainer;
        Evaluator *evaluator;

        if (marius_options.training.synchronous) {
            trainer = new SynchronousTrainer(train_set, model);
        } else {
            trainer = new PipelineTrainer(train_set, model);
        }

        if (will_evaluate) {
            if (marius_options.evaluation.synchronous) {
                evaluator = new SynchronousEvaluator(eval_set, model);
            } else {
                evaluator = new PipelineEvaluator(eval_set, model);
            }
        }

        for (int epoch = 0; epoch < marius_options.training.num_epochs; epoch += marius_options.evaluation.epochs_per_eval) {
            int num_epochs = marius_options.evaluation.epochs_per_eval;
            if (marius_options.training.num_epochs < num_epochs) {
                num_epochs = marius_options.training.num_epochs;
                trainer->train(num_epochs);
            } else {
                trainer->train(num_epochs);
                if (will_evaluate) {
                    evaluator->evaluate(epoch + marius_options.evaluation.epochs_per_eval < marius_options.training.num_epochs);
                }
            }
        }
        embeddings->unload(true);
        src_rel->unload(true);
        dst_rel->unload(true);


        // garbage collect
        delete trainer;
        delete train_set;
        if (will_evaluate) {
            delete evaluator;
            delete eval_set;
        }

        switch (marius_options.storage.edges) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                SPDLOG_ERROR("Backend type not available for edges.");
                exit(-1);
            }
            case BackendType::FlatFile: {
                delete (FlatFile *) train_edges;
                delete (FlatFile *) eval_edges;
                delete (FlatFile *) test_edges;
                break;
            }
            case BackendType::HostMemory: {
                delete (InMemory *) train_edges;
                delete (InMemory *) eval_edges;
                delete (InMemory *) test_edges;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) train_edges;
                delete (InMemory *) eval_edges;
                delete (InMemory *) test_edges;
                break;
            }
        }

        switch (marius_options.storage.embeddings) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                delete (PartitionBuffer *) embeddings;
                delete (PartitionBuffer *) emb_state;
                break;
            }
            case BackendType::FlatFile: {
                SPDLOG_ERROR("Backend type not available for embeddings.");
                exit(-1);
            }
            case BackendType::HostMemory: {
                delete (InMemory *) embeddings;
                delete (InMemory *) emb_state;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) embeddings;
                delete (InMemory *) emb_state;
                break;
            }
        }

        switch (marius_options.storage.relations) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                SPDLOG_ERROR("Backend type not available for relation embeddings.");
                exit(-1);
            }
            case BackendType::FlatFile: {
                SPDLOG_ERROR("Backend type not available for relation embeddings.");
                exit(-1);
            }
            case BackendType::HostMemory: {
                delete (InMemory *) embeddings;
                delete (InMemory *) emb_state;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) embeddings;
                delete (InMemory *) emb_state;
                break;
            }
        }
    } else {
        tuple<Storage *, Storage *, Storage *, Storage *> storage_ptrs = initializeEval();
        Storage *test_edges = get<0>(storage_ptrs);
        Storage *embeddings = get<1>(storage_ptrs);
        Storage *src_rel = get<2>(storage_ptrs);
        Storage *dst_rel = get<3>(storage_ptrs);

        eval_set = new DataSet(test_edges, embeddings, src_rel, dst_rel);

        preprocessing_timer.stop();
        int64_t preprocessing_time = preprocessing_timer.getDuration();

        SPDLOG_INFO("Preprocessing Complete: {}s", (double) preprocessing_time / 1000);

        Evaluator *evaluator;

        if (marius_options.evaluation.synchronous) {
            evaluator = new SynchronousEvaluator(eval_set, model);
        } else {
            evaluator = new PipelineEvaluator(eval_set, model);
        }
        evaluator->evaluate(false);

        delete eval_set;
        delete evaluator;

        switch (marius_options.storage.edges) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                SPDLOG_ERROR("Backend type not available for edges.");
                exit(-1);
            }
            case BackendType::FlatFile: {
                delete (FlatFile *) test_edges;
                break;
            }
            case BackendType::HostMemory: {
                delete (InMemory *) test_edges;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) test_edges;
                break;
            }
        }

        switch (marius_options.storage.embeddings) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                delete (PartitionBuffer *) embeddings;
                break;
            }
            case BackendType::FlatFile: {
                SPDLOG_ERROR("Backend type not available for embeddings.");
                exit(-1);
            }
            case BackendType::HostMemory: {
                delete (InMemory *) embeddings;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) embeddings;
                break;
            }
        }

        switch (marius_options.storage.relations) {
            case BackendType::RocksDB: {
                SPDLOG_ERROR("Currently Unsupported");
                exit(-1);
            }
            case BackendType::PartitionBuffer: {
                SPDLOG_ERROR("Backend type not available for relation embeddings.");
                exit(-1);
            }
            case BackendType::FlatFile: {
                SPDLOG_ERROR("Backend type not available for relation embeddings.");
                exit(-1);
            }
            case BackendType::HostMemory: {
                delete (InMemory *) embeddings;
                break;
            }
            case BackendType::DeviceMemory: {
                delete (InMemory *) embeddings;
                break;
            }
        }
    }

}

int main(int argc, char *argv[]) {
    marius(argc, argv);
}