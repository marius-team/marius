
#include "marius.h"

#include "common/util.h"
#include "configuration/util.h"
#include "pipeline/evaluator.h"
#include "pipeline/graph_encoder.h"
#include "pipeline/trainer.h"
#include "reporting/logger.h"
#include "storage/checkpointer.h"
#include "storage/io.h"

//#include "torch/torch.h"
//#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
//#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
//#include <torch/csrc/distributed/c10d/TCPStore.hpp>
//#include <torch/csrc/distributed/c10d/FileStore.hpp>
//#include <torch/csrc/distributed/c10d/Store.hpp>

void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<MariusConfig> marius_config) {
    shared_ptr<GraphEncoder> graph_encoder;
    if (marius_config->evaluation->pipeline->sync) {
        graph_encoder = std::make_shared<SynchronousGraphEncoder>(dataloader, model);
    } else {
        graph_encoder = std::make_shared<PipelineGraphEncoder>(dataloader, model, marius_config->evaluation->pipeline);
    }

    string filename = marius_config->storage->model_dir + PathConstants::encoded_nodes_file + PathConstants::file_ext;

    if (fileExists(filename)) {
        remove(filename.c_str());
    }

    int64_t num_nodes = marius_config->storage->dataset->num_nodes;

    int last_stage = marius_config->model->encoder->layers.size() - 1;
    int last_layer = marius_config->model->encoder->layers[last_stage].size() - 1;
    int64_t dim = marius_config->model->encoder->layers[last_stage][last_layer]->output_dim;

    dataloader->graph_storage_->storage_ptrs_.encoded_nodes = std::make_shared<FlatFile>(filename, num_nodes, dim, torch::kFloat32, true);

    graph_encoder->encode();
}

shared_ptr<c10d::ProcessGroupGloo> distributed_init(string coord_address, int world_size, int rank, string address) {

    auto store = c10::make_intrusive<c10d::TCPStore>(coord_address, 7654, world_size, rank==0);

    auto options = c10d::ProcessGroupGloo::Options::create();
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname(address));
//    options.timeout = std::chrono::milliseconds(1000);
//    options.threads =

    auto pg_gloo = std::make_shared<c10d::ProcessGroupGloo>(store, rank, world_size, options);

    return pg_gloo;


//    createDeviceForInterface
//    options->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice()); //this works
//    gloo::transport::tcp::attr attr;
//    options->devices.push_back(gloo::transport::tcp::CreateDevice(attr));

//    auto options1 = c10d::ProcessGroupNCCL::Options::create();
////    options->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());
////    options->is_high_priority_stream = false;
////    options->timeout = timeout;
//    auto pg1 = std::make_shared<c10d::ProcessGroupNCCL>(store, rank, 2, options1);


//    std::shared_ptr<::c10d::ProcessGroup::Work> work;
//    while (true) {
//        work = pg.allreduce(tensors);
//        try {
//            work->wait();

//    if (!work[i]->wait()) {
//        throw work[i]->exception();
//    }

//    /*numWorkerThreads=*/std::max(16U, std::thread::hardware_concurrency())


//    std::vector<torch::Tensor> vec;
//    torch::Tensor x = torch::randint(10, {5});
////    x = x.to(torch::Device(torch::kCUDA, 0));
//    std::cout<<"x: "<<x<<"\n";
//    vec.push_back(x);
//
//    //    pg.allreduce(vec);
//    auto work = pg->allreduce(vec);
////    auto work = pg1->allreduce(vec);
//    if (!work->wait()) {
//        throw work->exception();
//    }

}

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader>> marius_init(shared_ptr<MariusConfig> marius_config, bool train) {
    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    SPDLOG_INFO("Start initialization");

    MariusLogger marius_logger = MariusLogger(marius_config->storage->model_dir);
    spdlog::set_default_logger(marius_logger.main_logger_);
    marius_logger.setConsoleLogLevel(marius_config->storage->log_level);

    torch::manual_seed(marius_config->model->random_seed);
    srand(marius_config->model->random_seed);

    std::vector<torch::Device> devices = devices_from_config(marius_config->storage);

    shared_ptr<Model> model;
    shared_ptr<GraphModelStorage> graph_model_storage;

    int epochs_processed = 0;
    int num_partitions = 1; // TODO: move this to part of storage->dataset, can be set automatically during preprocessing
    if (marius_config->storage->embeddings != nullptr) {
        if (marius_config->storage->embeddings->type == StorageBackend::PARTITION_BUFFER) {
            num_partitions = std::dynamic_pointer_cast<PartitionBufferOptions>(marius_config->storage->embeddings->options)->num_partitions;
        }
    } else if (marius_config->storage->features != nullptr) {
        if (marius_config->storage->features->type == StorageBackend::PARTITION_BUFFER) {
            num_partitions = std::dynamic_pointer_cast<PartitionBufferOptions>(marius_config->storage->features->options)->num_partitions;
        }
    }

    if (train) {
        // initialize new model
        if (!marius_config->training->resume_training && marius_config->training->resume_from_checkpoint.empty()) {
            model = initModelFromConfig(marius_config->model, devices, marius_config->storage->dataset->num_relations, num_partitions, true);
            graph_model_storage = initializeStorage(model, marius_config->storage, !marius_config->training->resume_training, true);
        } else {
            auto checkpoint_loader = std::make_shared<Checkpointer>();

            string checkpoint_dir = marius_config->storage->model_dir;
            if (!marius_config->training->resume_from_checkpoint.empty()) {
                checkpoint_dir = marius_config->training->resume_from_checkpoint;
            }

            auto tup = checkpoint_loader->load(checkpoint_dir, marius_config, true);
            model = std::get<0>(tup);
            graph_model_storage = std::get<1>(tup);

            CheckpointMeta checkpoint_meta = std::get<2>(tup);
            epochs_processed = checkpoint_meta.num_epochs;
        }
    } else {
        auto checkpoint_loader = std::make_shared<Checkpointer>();

        string checkpoint_dir = marius_config->storage->model_dir;
        if (!marius_config->evaluation->checkpoint_dir.empty()) {
            checkpoint_dir = marius_config->evaluation->checkpoint_dir;
        }
        auto tup = checkpoint_loader->load(checkpoint_dir, marius_config, false);
        model = std::get<0>(tup);
        graph_model_storage = std::get<1>(tup);

        CheckpointMeta checkpoint_meta = std::get<2>(tup);
        epochs_processed = checkpoint_meta.num_epochs;
    }

    shared_ptr<DataLoader> dataloader = std::make_shared<DataLoader>(graph_model_storage, model->learning_task_, model->has_partition_embeddings(),
                                                                     marius_config->training, marius_config->evaluation, marius_config->model->encoder);

    dataloader->epochs_processed_ = epochs_processed;

    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();

    SPDLOG_INFO("Initialization Complete: {}s", (double)initialization_time / 1000);

    return std::forward_as_tuple(model, graph_model_storage, dataloader);
}

void marius_train(shared_ptr<MariusConfig> marius_config, shared_ptr<c10d::ProcessGroupGloo> pg_gloo) {
    auto tup = marius_init(marius_config, true);
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);

    if (pg_gloo != nullptr) {
        dataloader->setDistPG(pg_gloo, marius_config->distributed);
    }

    shared_ptr<Trainer> trainer;
    shared_ptr<Evaluator> evaluator;

    shared_ptr<Checkpointer> model_saver;
    CheckpointMeta metadata;
    if (marius_config->training->save_model) {
        model_saver = std::make_shared<Checkpointer>(model, graph_model_storage, marius_config->training->checkpoint);
        metadata.has_state = true;
        metadata.has_encoded = marius_config->storage->export_encoded_nodes;
        metadata.has_model = true;
        metadata.link_prediction = marius_config->model->learning_task == LearningTask::LINK_PREDICTION;
    }

    if (marius_config->training->pipeline->sync) {
        trainer = std::make_shared<SynchronousTrainer>(dataloader, model, marius_config->training->logs_per_epoch);
    } else {
        trainer = std::make_shared<PipelineTrainer>(dataloader, model, marius_config->training->pipeline, marius_config->training->logs_per_epoch);
    }

    if (marius_config->evaluation->pipeline->sync) {
        evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);
    } else {
        evaluator = std::make_shared<PipelineEvaluator>(dataloader, model, marius_config->evaluation->pipeline);
    }

    int checkpoint_interval = marius_config->training->checkpoint->interval;
    for (int epoch = 0; epoch < marius_config->training->num_epochs; epoch++) {
        trainer->train(1);

        if ((epoch + 1) % marius_config->evaluation->epochs_per_eval == 0) {
            if (marius_config->storage->dataset->num_valid != -1) {
                evaluator->evaluate(true);
            }

            if (marius_config->storage->dataset->num_test != -1) {
                evaluator->evaluate(false);
            }
        }

        metadata.num_epochs = dataloader->epochs_processed_;
        if (checkpoint_interval > 0 && (epoch + 1) % checkpoint_interval == 0 && epoch + 1 < marius_config->training->num_epochs) {
            model_saver->create_checkpoint(marius_config->storage->model_dir, metadata, dataloader->epochs_processed_);
        }
    }

    if (marius_config->training->save_model) {
        model_saver->save(marius_config->storage->model_dir, metadata);

        if (marius_config->storage->export_encoded_nodes) {
            encode_and_export(dataloader, model, marius_config);
        }
    }
}

void marius_eval(shared_ptr<MariusConfig> marius_config) {
    auto tup = marius_init(marius_config, false);
    auto model = std::get<0>(tup);
    auto graph_model_storage = std::get<1>(tup);
    auto dataloader = std::get<2>(tup);

    shared_ptr<Evaluator> evaluator;

    if (marius_config->evaluation->epochs_per_eval > 0) {
        if (marius_config->evaluation->pipeline->sync) {
            evaluator = std::make_shared<SynchronousEvaluator>(dataloader, model);
        } else {
            evaluator = std::make_shared<PipelineEvaluator>(dataloader, model, marius_config->evaluation->pipeline);
        }
        evaluator->evaluate(false);
    }

    if (marius_config->storage->export_encoded_nodes) {
        encode_and_export(dataloader, model, marius_config);
    }
}

void marius(int argc, char *argv[]) {
    (void)argc;

    bool train = true;
    string command_path = string(argv[0]);
    string config_path = string(argv[1]);
    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);
    if (strcmp(command_name.c_str(), "marius_eval") == 0) {
        train = false;
    }

    shared_ptr<c10d::ProcessGroupGloo> pg_gloo = nullptr;
    if (argc > 2) {
        SPDLOG_INFO("Distributed training detected.");

        string coord_address = string(argv[2]);
        int world_size = std::atoi(argv[3]);
        int rank = std::atoi(argv[4]);
        string address = string(argv[5]);

        pg_gloo = distributed_init(coord_address, world_size, rank, address);
    }

    shared_ptr<MariusConfig> marius_config = loadConfig(config_path, true);

    if (train) {
        marius_train(marius_config, pg_gloo);
    } else {
        marius_eval(marius_config);
    }
}

int main(int argc, char *argv[]) { marius(argc, argv); }