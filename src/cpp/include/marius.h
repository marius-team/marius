#include "configuration/config.h"
#include "data/dataloader.h"
#include "nn/model.h"
#include "storage/graph_storage.h"

void encode_and_export(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, shared_ptr<MariusConfig> marius_config);

std::tuple<shared_ptr<Model>, shared_ptr<GraphModelStorage>, shared_ptr<DataLoader> > marius_init(shared_ptr<MariusConfig> marius_config, bool train);

void marius_train(shared_ptr<MariusConfig> marius_config);

void marius_eval(shared_ptr<MariusConfig> marius_config);

void marius(int argc, char *argv[]);

int main(int argc, char *argv[]);
