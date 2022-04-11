//
// Created by Jason Mohoney on 2/9/22.
//

#include <gtest/gtest.h>
#include <data/samplers/negative.h>

int num_nodes = 6;

torch::Tensor edges = torch::tensor({{0, 2},
                                     {0, 4},
                                     {1, 3},
                                     {1, 5},
                                     {4, 2},
                                     {5, 2}}, torch::kInt64);

torch::Tensor typed_edges = torch::tensor({{0, 0, 2},
                                           {0, 1, 4},
                                           {1, 1, 3},
                                           {1, 0, 5},
                                           {4, 0, 2},
                                           {5, 1, 2}}, torch::kInt64);

torch::Tensor batch_edges = torch::tensor({{1, 5}, {0, 2}, {4, 2}}, torch::kInt64);
torch::Tensor batch_typed_edges = torch::tensor({{1, 0, 5}, {0, 0, 2}, {4, 0, 2}}, torch::kInt64);

class CorruptNodeNegativeSamplerTest : public ::testing::Test {
protected:
    shared_ptr<MariusGraph> graph;
    shared_ptr<MariusGraph> typed_graph;

    void SetUp() override {
        torch::Tensor dst_sorted_edges = edges.index_select(0, edges.select(1, 1).argsort(0));
        torch::Tensor dst_sorted_typed_edges = typed_edges.index_select(0, typed_edges.select(1, 2).argsort(0));

        graph = std::make_shared<MariusGraph>(edges, dst_sorted_edges, num_nodes);
        typed_graph = std::make_shared<MariusGraph>(typed_edges, dst_sorted_typed_edges, num_nodes);

        graph->sortAllEdges(torch::tensor({{0, 3}}, torch::kInt64));
        typed_graph->sortAllEdges(torch::tensor({{0, 1, 3}}, torch::kInt64));
    }
};

void validate_sample(shared_ptr<CorruptNodeNegativeSampler> sampler, torch::Tensor sample, shared_ptr<MariusGraph> graph) {

    // validate shape
    ASSERT_EQ(sample.size(0), sampler->num_chunks_);

    if (sampler->num_negatives_ != -1) {
        ASSERT_EQ(sample.size(1), sampler->num_negatives_);
    } else {
        ASSERT_EQ(sample.size(1), graph->num_nodes_in_memory_);
    }

    // validate max and min ids
    ASSERT_TRUE(sample.max().item<int64_t>() < graph->num_nodes_in_memory_);
    ASSERT_TRUE(sample.min().item<int64_t>() >= 0);
}

void validate_filter_local(torch::Tensor filter, torch::Tensor sample, torch::Tensor edges_t, bool inverse) {

    // check filtered edges are present in the graph
    auto batch_accessor = edges_t.accessor<int64_t, 2>();
    auto sample_accessor = sample.accessor<int64_t, 2>();
    auto filter_accessor = filter.accessor<int64_t, 2>();

    bool has_relations = false;
    if (edges_t.size(1) == 3) {
        has_relations = true;
    }

    int64_t num_chunks = sample.size(0);
    int64_t num_edges = edges_t.size(0);
    int64_t chunk_size = ceil((double) num_edges / num_chunks);

    for (int i = 0; i < filter.size(0); i++) {
        int64_t src;
        int64_t rel;
        int64_t dst;

        bool found = false;

        int64_t edge_id = filter_accessor[i][0];

        int chunk_id = edge_id / chunk_size;

        if (inverse) {
            src = sample_accessor[chunk_id][filter_accessor[i][1]];
            if (has_relations) {
                rel = batch_accessor[edge_id][1];
                dst = batch_accessor[edge_id][2];
            } else {
                dst = batch_accessor[edge_id][1];
            }
        } else {
            src = batch_accessor[edge_id][0];
            dst = sample_accessor[chunk_id][filter_accessor[i][1]];
            if (has_relations) {
                rel = batch_accessor[edge_id][1];
            }
        }

        if (has_relations) {
            for (int k = 0; k < edges_t.size(0); k++) {
                if (batch_accessor[k][0] == src && batch_accessor[k][1] == rel && batch_accessor[k][2] == dst) {
                    found = true;
                }
            }
        } else {
            for (int k = 0; k < edges_t.size(0); k++) {
                if (batch_accessor[k][0] == src && batch_accessor[k][1] == dst) {
                    found = true;
                }
            }
        }
        ASSERT_TRUE(found);
    }
}

void validate_filter_global(torch::Tensor filter, torch::Tensor sample, shared_ptr<MariusGraph> graph, torch::Tensor edges_t, bool inverse) {

    // check filtered edges are present in the graph
    auto graph_accessor = graph->src_sorted_edges_.accessor<int64_t, 2>();
    auto batch_accessor = edges_t.accessor<int64_t, 2>();
    auto sample_accessor = sample.accessor<int64_t, 2>();
    auto filter_accessor = filter.accessor<int64_t, 2>();

    bool has_relations = false;
    if (edges_t.size(1) == 3) {
        has_relations = true;
    }

    int64_t num_chunks = sample.size(0);
    int64_t num_edges =  edges_t.size(0);
    int64_t chunk_size = ceil((double) num_edges / num_chunks);

    for (int i = 0; i < filter.size(0); i++) {
        int64_t src;
        int64_t rel;
        int64_t dst;

        bool found = false;

        int64_t edge_id = filter_accessor[i][0];

        int chunk_id = edge_id / chunk_size;

        if (inverse) {
            src = sample_accessor[chunk_id][filter_accessor[i][1]];
            if (has_relations) {
                rel = batch_accessor[edge_id][1];
                dst = batch_accessor[edge_id][2];
            } else {
                dst = batch_accessor[edge_id][1];
            }
        } else {
            src = batch_accessor[edge_id][0];
            dst = sample_accessor[chunk_id][filter_accessor[i][1]];
            if (has_relations) {
                rel = batch_accessor[edge_id][1];
            }
        }

        if (has_relations) {
            for (int k = 0; k < graph->src_sorted_edges_.size(0); k++) {
                if (graph_accessor[k][0] == src && graph_accessor[k][1] == rel && graph_accessor[k][2] == dst) {
                    found = true;
                }
            }
        } else {
            for (int k = 0; k < graph->src_sorted_edges_.size(0); k++) {
                if (graph_accessor[k][0] == src && graph_accessor[k][1] == dst) {
                    found = true;
                }
            }
        }
        ASSERT_TRUE(found);
    }
}

void test_unfiltered_corruption_sampler(shared_ptr<CorruptNodeNegativeSampler> sampler, shared_ptr<MariusGraph> graph, torch::Tensor edges_t) {
    torch::Tensor sample;
    torch::Tensor filter;

    std::tie(sample, filter) = sampler->getNegatives(graph, edges_t, false);
    validate_sample(sampler, sample, graph);
    validate_filter_local(filter, sample, edges_t, false);

    std::tie(sample, filter) = sampler->getNegatives(graph, edges_t, true);
    validate_sample(sampler, sample, graph);
    validate_filter_local(filter, sample, edges_t, true);
}

void test_filtered_corruption_sampler(shared_ptr<CorruptNodeNegativeSampler> sampler, shared_ptr<MariusGraph> graph, torch::Tensor edges_t) {
    torch::Tensor sample;
    torch::Tensor filter;

    std::tie(sample, filter) = sampler->getNegatives(graph, edges_t, false);
    validate_sample(sampler, sample, graph);
    validate_filter_global(filter, sample, graph, edges_t, false);

    std::tie(sample, filter) = sampler->getNegatives(graph, edges_t, true);
    validate_sample(sampler, sample, graph);
    validate_filter_global(filter, sample, graph, edges_t, true);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestUniform) {
    auto corrupt_uniform = std::make_shared<CorruptNodeNegativeSampler>(1, 5, 0.0, false);
    test_unfiltered_corruption_sampler(corrupt_uniform, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_uniform, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestUniformChunked) {
    auto corrupt_uniform_chunked = std::make_shared<CorruptNodeNegativeSampler>(3, 5, 0.0, false);
    test_unfiltered_corruption_sampler(corrupt_uniform_chunked, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_uniform_chunked, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestMix) {
    auto corrupt_mix = std::make_shared<CorruptNodeNegativeSampler>(1, 5, 0.5, false);
    test_unfiltered_corruption_sampler(corrupt_mix, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_mix, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestMixChunked) {
    auto corrupt_mix_chunked = std::make_shared<CorruptNodeNegativeSampler>(3, 5, 0.5, false);
    test_unfiltered_corruption_sampler(corrupt_mix_chunked, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_mix_chunked, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestAllDegree) {
    auto corrupt_all_degree = std::make_shared<CorruptNodeNegativeSampler>(1, 5, 1.0, false);
    test_unfiltered_corruption_sampler(corrupt_all_degree, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_all_degree, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestAllDegreeChunked) {
    auto corrupt_all_degree_chunked = std::make_shared<CorruptNodeNegativeSampler>(3, 5, 1.0, false);
    test_unfiltered_corruption_sampler(corrupt_all_degree_chunked, graph, batch_edges);
    test_unfiltered_corruption_sampler(corrupt_all_degree_chunked, typed_graph, batch_typed_edges);
}

TEST_F(CorruptNodeNegativeSamplerTest, TestFilter) {
    auto corrupt_filtered = std::make_shared<CorruptNodeNegativeSampler>(1, -1, 0.0, true);
    test_filtered_corruption_sampler(corrupt_filtered, graph, batch_edges);
    test_filtered_corruption_sampler(corrupt_filtered, typed_graph, batch_typed_edges);
}