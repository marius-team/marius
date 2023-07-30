#include "configuration/config.h"
#include "data/dataloader.h"
#include "nn/model.h"
#include "storage/graph_storage.h"

#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<MariusConfig> marius_config);

shared_ptr<c10d::ProcessGroupGloo> distributed_init(string coord_address, int world_size, int rank, string address);

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader>> marius_init(shared_ptr<MariusConfig> marius_config, bool train);

void marius_train(shared_ptr<MariusConfig> marius_config, shared_ptr<c10d::ProcessGroupGloo> pg_gloo = nullptr);

void marius_eval(shared_ptr<MariusConfig> marius_config);

void marius(int argc, char *argv[]);

int main(int argc, char *argv[]);
