//
// Created by Jason Mohoney on 4/12/21.
//

#include <gtest/gtest.h>
#include <config.h>
#include <string>

std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";


TEST(TestConfig, TestConfigOnly) {
    // Test that the default config works properly
    int num_args = 2;
    const char* n_argv[] = {"marius_train", conf_str.c_str()};
    parseConfig(num_args, (char **)n_argv);
}

TEST(TestConfig, TestCommandLineOverride) {
    // Test that command line options override the config file options
    int num_args = 8;
    const char* n_argv[] = {"marius_train", conf_str.c_str(), 
    "--general.random_seed=5", "--model.scale_factor=.003",
    "--evaluation.epochs_per_eval=3", "--evaluation.filtered_evaluation=true",
    "--path.node_ids=override_path", "--general.gpu_ids=0,1,2,3"};
    marius_options = parseConfig(num_args, (char **)n_argv);
    EXPECT_EQ(marius_options.general.random_seed,5);
    EXPECT_FLOAT_EQ(marius_options.model.scale_factor,.003);
    EXPECT_EQ(marius_options.evaluation.epochs_per_eval,3);
    EXPECT_EQ(marius_options.evaluation.filtered_evaluation,true);
    EXPECT_EQ(marius_options.path.node_ids,"override_path");
    EXPECT_EQ(marius_options.general.gpu_ids[0],0);
    EXPECT_EQ(marius_options.general.gpu_ids[1],1);
    EXPECT_EQ(marius_options.general.gpu_ids[2],2);
    EXPECT_EQ(marius_options.general.gpu_ids[3],3);
}

TEST(TestConfig, TestInvalidConfig) {
    // Test that invalid configuration option names fail
    int num_args = 3;
    const char* n_argv1[] = {"marius_train", conf_str.c_str(), "bad_argument"};
    EXPECT_DEATH(parseConfig(num_args, (char **)n_argv1), "");
    const char* n_argv2[] = {"marius_train", conf_str.c_str(), "--model."};
    EXPECT_DEATH(parseConfig(num_args, (char **)n_argv2), "");
    const char* n_argv3[] = {"marius_train", conf_str.c_str(), "--model.fake_option=true"};
    EXPECT_DEATH(parseConfig(num_args, (char **)n_argv3), "");
}

TEST(TestConfig, TestMissingConfig) {
    // test exception is thrown when configuration file is missing
    int num_args1 = 1;
    const char* n_argv1[] = {"marius_train"};
    EXPECT_DEATH(parseConfig(num_args1, (char **)n_argv1), "");
    int num_args2 = 2;
    const char* n_argv2[] = {"marius_train", "bad_file.ini"};
    EXPECT_DEATH(parseConfig(num_args2, (char **)n_argv2), "");
}

TEST(TestConfig, TestInvalidNumericalOptions) {
    // test numerical options fail when out of range
    int num_args1 = 3;
    const char* n_argv1[] = {"marius_train", conf_str.c_str(), "--general.random_seed=-1"};
    EXPECT_DEATH(parseConfig(num_args1, (char **)n_argv1), "");
    int num_args2 = 3;
    const char* n_argv2[] = {"marius_train", conf_str.c_str(), "--evaluation.degree_fraction=2.0"};
    EXPECT_DEATH(parseConfig(num_args2, (char **)n_argv2), "");
}

TEST(TestConfig, TestInvalidEnumString) {
    // test that error/exception occurs when invalid strings are used for enum values
    int num_args = 3;
    const char* n_argv[] = {"marius_train", conf_str.c_str(), "--model.initialization_distribution=invalid"};
    EXPECT_DEATH(parseConfig(num_args, (char **)n_argv), "");
}

TEST(TestConfig, TestHelp) {
    // test help messages display when flag set
    int num_args = 3;
    testing::internal::CaptureStdout();
    const char* n_argv1[] = {"marius_train", conf_str.c_str(), "-h"};
    EXPECT_EXIT(parseConfig(num_args, (char **)n_argv1), ::testing::ExitedWithCode(0), "");
    std::string output = testing::internal::GetCapturedStdout();
    if (output.find("Usage") == std::string::npos)
        FAIL();
    testing::internal::CaptureStdout();
    const char* n_argv2[] = {"marius_train", conf_str.c_str(), "--help"};
    EXPECT_EXIT(parseConfig(num_args, (char **)n_argv2), ::testing::ExitedWithCode(0), "");
    output = testing::internal::GetCapturedStdout();
    if (output.find("Usage") == std::string::npos)
        FAIL();
}

TEST(TestConfig, TestHelpOnInvalidInput) {
    // test help messages display with bad input
    int num_args = 3;
    testing::internal::CaptureStdout();
    const char* n_argv[] = {"marius_train", conf_str.c_str(), "bad_argument"};
    EXPECT_DEATH(parseConfig(num_args, (char **)n_argv), "");
    std::string output = testing::internal::GetCapturedStdout();
    if (output.find("Usage") == std::string::npos)
        FAIL();
}