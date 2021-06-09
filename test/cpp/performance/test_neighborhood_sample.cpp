//
// Created by Jason Mohoney on 6/9/21.
//

#include <gtest/gtest.h>

#include "config.h"
#include "storage.h"
#include "util.h"

TEST(TestNeighborhoodSample, TestRuntimeFB15k) {
    if (!std::getenv("RUN_PERFORMANCE")) {
        GTEST_SKIP();
    }

    string filename = "output_dir/train_edges.pt";
    string edges_partitions = "output_dir/train_edges_partitions.txt";
    int dim0_size = 483142;
    int dim1_size = 3;
    torch::ScalarType dtype = torch::kInt32;

    std::vector<int> init_buffer = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // set the required options
    MariusOptions options;
    StorageOptions storage_options;
    storage_options.num_partitions = 16;
    options.storage = storage_options;
    marius_options = options;

    FlatFile *fb15k_edges = new FlatFile(filename, dim0_size, dim1_size, dtype);
    fb15k_edges->readPartitionSizes(edges_partitions);
    fb15k_edges->load();
    fb15k_edges->initializeInMemorySubGraph(init_buffer);


    Timer timer = Timer(false);
    for (int i = 1; i <= 10; i++) {
        torch::Tensor test_id = torch::randint(0, 14951, i * 1000);

        timer.start();
        auto ret = fb15k_edges->gatherNeighbors(test_id, true);
        auto out_ids = std::get<0>(ret);
        auto out_offsets = std::get<1>(ret);

        ret = fb15k_edges->gatherNeighbors(test_id, false);
        auto in_ids = std::get<0>(ret);
        auto in_offsets = std::get<1>(ret);

        timer.stop();

        std::cout << "n = " << i * 1000 << ", t = " << timer.getDuration() << "ms\n";
    }



//    auto unique_nodes = std::get<0>(torch::_unique(torch::cat({in_ids.select(1, 0), in_ids.select(1, 2), out_ids.select(1, 0), out_ids.select(1, 2)})));
//    std::cout << unique_nodes.size(0) << '\n';
}