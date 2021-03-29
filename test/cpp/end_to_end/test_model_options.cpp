//
// Created by Jason Mohoney on 3/28/21.
//

#include <gtest/gtest.h>
#include <marius.h>


TEST(TestModelOptions, TestDecoderModel) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
    const char* conf = conf_str.c_str();

    int num_args = 3;
    const char* n_argv1[] = {"marius_train", conf,  "--model.decoder=DistMult"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "--model.decoder=ComplEx"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "--model.decoder=TransE"};
    marius(num_args, (char **)(n_argv3));
}
