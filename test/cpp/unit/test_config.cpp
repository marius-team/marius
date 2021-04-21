//
// Created by Jason Mohoney on 4/12/21.
//

#include <config.h>
#include <string>

std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";


TEST(TestConfig, TestConfigOnly) {
    // Test that the default config works properly
}

TEST(TestConfig, TestCommandLineOverride) {
    // Test that command line options override the config file options
}

TEST(TestConfig, TestInvalidConfig) {
    // Test that invalid configuration option names fail
}

TEST(TestConfig, TestMissingConfig) {
    // test exception is thrown when configuration file is missing
}

TEST(TestConfig, TestInvalidNumericalOptions) {
    // test numerical options fail when out of range
}

TEST(TestConfig, TestInvalidEnumString) {
    // test that error/exception occurs when invalid strings are used for enum values
}
