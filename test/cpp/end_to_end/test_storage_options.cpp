//
// Created by Jason Mohoney on 3/28/21.
//

#include <gtest/gtest.h>
#include <marius.h>

TEST(TestStorageOptions, TestEdgesBackend) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
    const char* conf = conf_str.c_str();

    int num_args = 3;
    const char* n_argv1[] = {"marius_train", conf,  "--storage.edges_backend=HostMemory"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "--storage.edges_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "--storage.edges_backend=FlatFile"};
    marius(num_args, (char **)(n_argv3));

}

TEST(TestStorageOptions, TestEmbeddingsBackend) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
    const char* conf = conf_str.c_str();

    int num_args = 3;
    const char* n_argv1[] = {"marius_train", conf,  "--storage.embeddings_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "--storage.embeddings_backend=HostMemory"};
    marius(num_args, (char **)(n_argv2));
}

TEST(TestStorageOptions, TestRelationsBackend) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
    const char* conf = conf_str.c_str();

    int num_args = 3;
    const char* n_argv1[] = {"marius_train", conf,  "--storage.relations_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "--storage.relations_backend=HostMemory"};
    marius(num_args, (char **)(n_argv2));
}