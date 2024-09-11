#include <iostream>
#include <chrono>
#include <cstring>
#include <numa.h>
#include <simsimd/simsimd.h>
#include <fcntl.h>
#include <unistd.h>


// Define the necessary constants
const int vector_size = 128;
const size_t ONE_GB = 1024 * 1024 * 1024;  
const float NUM_CHUNKS = 0.1;
const size_t BUFFER_SIZE = NUM_CHUNKS * ONE_GB; // Buffer size of 10 GB

void benchmark_scan_list(float* query_vec, float* search_vectors, size_t num_vectors, std::string benchmark_name) {
    float total_distance = 0.0;
    double dist_result;

    // Run the actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_vectors; i++) {
        float* curr_search_vec = search_vectors + i * vector_size;
        simsimd_dot_f32(query_vec, curr_search_vec, vector_size, &dist_result);
        total_distance += dist_result;
    }
    auto end = std::chrono::high_resolution_clock::now();
    float time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Benchmark " << benchmark_name << ": Got average distance of " << total_distance/num_vectors << " at throughput of " << num_vectors/time_taken << " vectors/ms" << std::endl;
}

int main() {
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system" << std::endl;
        return 1;
    }

    // Ensure we have at least 2 NUMA nodes
    if (numa_max_node() < 1) {
        std::cerr << "This system doesn't have at least 2 NUMA nodes" << std::endl;
        return 1;
    }

    if(numa_run_on_node(0) != 0) {
        std::cerr << "Failed to force worker to run on node 0" << std::endl;
    }

    int random_fd = open("/dev/random", O_RDONLY);

    // Initialize the query vector
    float query_vector[vector_size];
    read(random_fd, query_vector, vector_size * sizeof(float));

    // Create the target vector on both nodes
    float* node_zero_vectors = reinterpret_cast<float*>(numa_alloc_onnode(BUFFER_SIZE, 0));
    float* node_one_vectors = reinterpret_cast<float*>(numa_alloc_onnode(BUFFER_SIZE, 1));
    if (!node_zero_vectors || !node_one_vectors) {
        std::cerr << "Failed to allocate memory on numa nodes" << std::endl;
        numa_free(node_zero_vectors, BUFFER_SIZE);
        numa_free(node_one_vectors, BUFFER_SIZE);
        close(random_fd);
        return 1;
    }
    size_t single_vector_size = vector_size * sizeof(float);
    size_t num_vectors = BUFFER_SIZE/single_vector_size;

    // Populate both the buffers with data
    read(random_fd, node_one_vectors, BUFFER_SIZE);
    read(random_fd, node_zero_vectors, BUFFER_SIZE);

    // Run the cross node benchmark
    std::cout << "Running benchmarking for buffer size of " << NUM_CHUNKS << " GB" << std::endl;
    benchmark_scan_list(query_vector, node_one_vectors, num_vectors, "Node1 Vectors");
    numa_free(node_one_vectors, BUFFER_SIZE);

    // Run the same node benchmark
    benchmark_scan_list(query_vector, node_zero_vectors, num_vectors, "Node0 Vectors");
    numa_free(node_zero_vectors, BUFFER_SIZE);
    close(random_fd);

    return 0;
}