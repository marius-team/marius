
#include "marius.h"

#include "configuration/config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"

void marius(int argc, char *argv[]) {

    bool train = true;
    string command_path = string(argv[0]);
    string config_path = string(argv[1]);
    string command_name = command_path.substr(command_path.find_last_of("/\\") + 1);
    if (strcmp(command_name.c_str(), "marius_eval") == 0) {
        train = false;
    }

    shared_ptr<MariusConfig> marius_config = initConfig(config_path);

    torch::manual_seed(marius_config->model->random_seed);
    srand(marius_config->model->random_seed);

    Timer initialization_timer = Timer(false);
    initialization_timer.start();
    SPDLOG_INFO("Start initialization");

    std::vector<torch::Device> devices;

    if (marius_config->storage->device_type == torch::kCUDA) {
        for (int i = 0; i < marius_config->storage->device_ids.size(); i++) {
            devices.emplace_back(torch::Device(torch::kCUDA, marius_config->storage->device_ids[i]));
        }

        if (devices.empty()) {
            devices.emplace_back(torch::Device(torch::kCUDA, 0));
        }
    } else {
        devices.emplace_back(torch::kCPU);
    }

    std::shared_ptr<Model> model = initializeModel(marius_config->model,
                                                   devices,
                                                   marius_config->storage->dataset->num_relations);
    model->train_ = train;

    if (marius_config->evaluation->negative_sampling != nullptr) {
        model->filtered_eval_ = marius_config->evaluation->negative_sampling->filtered;
    } else {
        model->filtered_eval_ = false;
    }

    GraphModelStorage *graph_model_storage = initializeStorage(model, marius_config->storage);

    DataLoader *dataloader = new DataLoader(graph_model_storage,
                                            marius_config->training,
                                            marius_config->evaluation,
                                            marius_config->model->encoder);

    initialization_timer.stop();
    int64_t initialization_time = initialization_timer.getDuration();

    SPDLOG_INFO("Initialization Complete: {}s", (double) initialization_time / 1000);

    Trainer *trainer;
    Evaluator *evaluator;

    if (train) {
        if (marius_config->training->pipeline->sync) {
            if (marius_config->storage->device_ids.size() > 1) {
                trainer = new SynchronousMultiGPUTrainer(dataloader, model, marius_config->training->logs_per_epoch);
            } else {
                trainer = new SynchronousTrainer(dataloader, model, marius_config->training->logs_per_epoch);
            }
        } else {
            trainer = new PipelineTrainer(dataloader,
                                          model,
                                          marius_config->training->pipeline,
                                          marius_config->training->logs_per_epoch);
        }

        if (marius_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              marius_config->evaluation->pipeline);
        }

        for (int epoch = 0; epoch < marius_config->training->num_epochs; epoch++) {
            if ((epoch + 1) % marius_config->evaluation->epochs_per_eval != 0) {
                trainer->train(1);
            } else {
                trainer->train(1);
                evaluator->evaluate(true);
                evaluator->evaluate(false);
            }
        }
    } else {
        if (marius_config->evaluation->pipeline->sync) {
            evaluator = new SynchronousEvaluator(dataloader, model);
        } else {
            evaluator = new PipelineEvaluator(dataloader,
                                              model,
                                              marius_config->evaluation->pipeline);
        }
        evaluator->evaluate(false);
    }

    model->save(marius_config->storage->dataset->base_directory);

    // garbage collect
    delete graph_model_storage;
    delete trainer;
    delete evaluator;
    delete dataloader;
}

int main(int argc, char *argv[]) {
    marius(argc, argv);
}