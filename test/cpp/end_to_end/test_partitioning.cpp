//
// Created by Jason Mohoney on 3/28/21.
//

#include <gtest/gtest.h>
#include <marius.h>

TEST(TestPartitioning, TestN10) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/test_partitioning.ini";
    const char* conf = conf_str.c_str();

    // Need a better way to test the partitioning code end to end
//    int num_args = 3;
//    const char* n_argv1[] = {"marius_train", conf};
//    marius(num_args, (char **)(n_argv1));
}
