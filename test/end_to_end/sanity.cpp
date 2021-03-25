//
// Created by Jason Mohoney on 10/19/20.
//

#include <gtest/gtest.h>
#include <marius.h>
#include <string>


std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
const char* conf = conf_str.c_str();

/**
 * Runs marius training on a default test configuration
 */
TEST(TestMain, TestDefaultConfig) {
    int num_args = 3;
    const char* n_argv[] = {"marius_train", conf,  "info"};
    marius(num_args, (char **)(n_argv));
}

// GENERAL

// device
TEST(TestMain, TestDevice) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--general.device=CPU"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--general.device=GPU"};
    marius(num_args, (char **)(n_argv2));
}

// comparator_type
TEST(TestMain, TestComparatorType) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--general.comparator_type=Cosine"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--general.comparator_type=Dot"};
    marius(num_args, (char **)(n_argv2));
}

// relation_type
TEST(TestMain, TestRelationType) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--general.relation_type=Translation"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--general.relation_type=Hadamard"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--general.relation_type=ComplexHadamard"};
    marius(num_args, (char **)(n_argv3));
    const char* n_argv4[] = {"marius_train", conf,  "info", "--general.relation_type=NoOp"};
    marius(num_args, (char **)(n_argv4));
}

// STORAGE

// edges_backend
TEST(TestMain, TestEdgesBackend) {
    int num_args = 4;
    // const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.edges_backend=RocksDB"};
    // marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.edges_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--storage.edges_backend=FlatFile"};
    marius(num_args, (char **)(n_argv3));
    const char* n_argv4[] = {"marius_train", conf,  "info", "--storage.edges_backend=HostMemory"};
    marius(num_args, (char **)(n_argv4));
}

// reinit_edges
TEST(TestMain, TestReinitEdges) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.reinit_edges=true"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.reinit_edges=false"};
    marius(num_args, (char **)(n_argv2));
}

// breaks above

// embeddings_backend
TEST(TestMain, TestEmbeddingsBackend) {
    int num_args = 4;
    //const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.edges_backend=RocksDB"};
    //marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=HostMemory"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv3));
    //const char* n_argv4[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=FlatFile"};
    //marius(num_args, (char **)(n_argv4));
    //const char* n_argv5[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=PartitionBuffer"};
    //marius(num_args, (char **)(n_argv5));
}

// reinit_embeddings
TEST(TestMain, TestReinitEmbeddings) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.reinit_embeddings=true"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.reinit_embeddings=false"};
    marius(num_args, (char **)(n_argv2));
}

// relations_backend
TEST(TestMain, TestRelationsBackend) {
    int num_args = 4;
    //const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.edges_backend=RocksDB"};
    //marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.relations_backend=DeviceMemory"};
    marius(num_args, (char **)(n_argv2));
    //const char* n_argv3[] = {"marius_train", conf,  "info", "--storage.relations_backend=FlatFile"};
    //marius(num_args, (char **)(n_argv3));
    const char* n_argv4[] = {"marius_train", conf,  "info", "--storage.relations_backend=HostMemory"};
    marius(num_args, (char **)(n_argv4));
}

// prefetching
TEST(TestMain, TestPrefetching) {
    int num_args = 4;
    //const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=PartitionBuffer --storage.num_partitions=5 --storage.buffer_capacity=3 --storage.prefetching=true"};
    //marius(num_args, (char **)(n_argv1));
    const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.prefetching=true"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.prefetching=false"};
    marius(num_args, (char **)(n_argv2));
}

// conserve_memory
TEST(TestMain, TestConserveMemory) {
    int num_args = 4;
    // const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.embeddings_backend=PartitionBuffer --storage.conserve_memory=true"};
    // marius(num_args, (char **)(n_argv1));
    const char* n_argv1[] = {"marius_train", conf,  "info", "--storage.conserve_memory=true"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--storage.conserve_memory=false"};
    marius(num_args, (char **)(n_argv2));
}

// // TRAINING

// optimizer_type
TEST(TestMain, TestOptimizerType) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training.optimizer_type=SGD"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.optimizer_type=Adagrad"};
    marius(num_args, (char **)(n_argv2));
}

// loss
TEST(TestMain, TestLoss) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training.loss=Ranking"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.loss=SoftMax"};
    marius(num_args, (char **)(n_argv2));
}

// negative_sampling_access
TEST(TestMain, TestNegSampAccess) {
    int num_args = 4;
    //computer might be slow
    //const char* n_argv1[] = {"marius_train", conf,  "info", "--training.negative_sampling_access=RandomSequential"};
    //marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.negative_sampling_access=SequentialSample"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--training.negative_sampling_access=Uniform"};
    marius(num_args, (char **)(n_argv3));
}

// negative_sampling_policy
TEST(TestMain, TestNegSampPolicy) {
    // NoReuse didn't fail!
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training.negative_sampling_policy=BatchReuse"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.negative_sampling_policy=DegreeBased"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--training.negative_sampling_policy=NoReuse"};
    marius(num_args, (char **)(n_argv3));
}

// edge_bucket_ordering
TEST(TestMain, TestEdgeBucketOrdering) {
    int num_args = 4;
    // const char* n_argv1[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=Hilbert"};
    // marius(num_args, (char **)(n_argv1));
    // const char* n_argv2[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=HilbertSymmetric"};
    // marius(num_args, (char **)(n_argv2));
    // const char* n_argv3[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=Random"};
    // marius(num_args, (char **)(n_argv3));
    // const char* n_argv4[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=RandomSymmetric"};
    // marius(num_args, (char **)(n_argv4));
    // const char* n_argv5[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=Sequential"};
    // marius(num_args, (char **)(n_argv5));
    // const char* n_argv6[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=SequentialSymmetric"};
    // marius(num_args, (char **)(n_argv6));
    // const char* n_argv7[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=Elimination"};
    // marius(num_args, (char **)(n_argv7));
    const char* n_argv8[] = {"marius_train", conf,  "info", "--training.edge_bucket_ordering=Shuffle"};
    marius(num_args, (char **)(n_argv8));
}

// average_gradients
TEST(TestMain, TestAvgGradients) {
    // Did not fail! No option?
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training.average_gradients=true"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.average_gradients=false"};
    marius(num_args, (char **)(n_argv2));
}

// synchronous
TEST(TestMain, TestAsync) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training.synchronous=false"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training.synchronous=true"};
    marius(num_args, (char **)(n_argv2));
}

// TRAINING PIPELINE

// update_in_flight
TEST(TestMain, TestUpdateInFlight) {
    // Did not fail, no option
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--training_pipeline.update_in_flight=false"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--training_pipeline.update_in_flight=true"};
    marius(num_args, (char **)(n_argv2));
}

// EVALUATION

// negative_sampling_access
TEST(TestMain, TestNegSampAccessEval) {
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--evaluation.negative_sampling_access=Uniform"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--evaluation.negative_sampling_access=SequentialSample"};
    marius(num_args, (char **)(n_argv2));
}

// negative_sampling_policy
TEST(TestMain, TestNegSampPolicyEval) {
    // NoReuse didn't fail!
    int num_args = 4;
    const char* n_argv1[] = {"marius_train", conf,  "info", "--evaluation.negative_sampling_policy=BatchReuse"};
    marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--evaluation.negative_sampling_policy=DegreeBased"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--evaluation.negative_sampling_policy=NoReuse"};
    marius(num_args, (char **)(n_argv3));
}

// evaluation_method
TEST(TestMain, TestEvalMethod) {
    int num_args = 4;
    //const char* n_argv1[] = {"marius_train", conf,  "info", "--evaluation.evaluation_method=FilteredMRR"};
    //marius(num_args, (char **)(n_argv1));
    const char* n_argv2[] = {"marius_train", conf,  "info", "--evaluation.evaluation_method=NodeClassification"};
    marius(num_args, (char **)(n_argv2));
    const char* n_argv3[] = {"marius_train", conf,  "info", "--evaluation.evaluation_method=MRR"};
    marius(num_args, (char **)(n_argv3));
}