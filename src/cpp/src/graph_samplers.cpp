//
// Created by Jason Mohoney on 8/25/21.
//

#include "graph_samplers.h"

//#include <unordered_set>
#include <parallel_hashmap/phmap.h>

#ifdef MARIUS_OMP
    #include "omp.h"
#endif

RandomEdgeSampler::RandomEdgeSampler(GraphModelStorage *graph_storage, bool without_replacement) {
    graph_storage_ = graph_storage;
    without_replacement_ = without_replacement;
}

EdgeList RandomEdgeSampler::getEdges(Batch *batch) {
    return graph_storage_->getEdgesRange(batch->start_idx_, batch->batch_size_).clone().to(torch::kInt64);
}

void NegativeSampler::lock() {
    sampler_lock_.lock();
}

void NegativeSampler::unlock() {
    sampler_lock_.unlock();
}

RandomNegativeSampler::RandomNegativeSampler(GraphModelStorage *graph_storage,
                                             int num_chunks,
                                             int num_negatives,
                                             bool without_replacement) {
    graph_storage_ = graph_storage;
    num_chunks_ = num_chunks;
    num_negatives_ = num_negatives;
    without_replacement_ = without_replacement;
}

torch::Tensor RandomNegativeSampler::getNegatives(Batch *batch, bool src) {
    vector<Indices> ret_indices(num_chunks_);

    for (int j = 0; j < num_chunks_; j++) {
        ret_indices[j] = graph_storage_->getRandomNodeIds(num_negatives_);
    }

    return torch::stack(ret_indices).flatten(0, 1);
}

FilteredNegativeSampler::FilteredNegativeSampler(GraphModelStorage *graph_storage) {
    graph_storage_ = graph_storage;
}

torch::Tensor FilteredNegativeSampler::getNegatives(Batch *batch, bool src) {
    return torch::arange(graph_storage_->getNumNodes());
}

LayeredNeighborSampler::LayeredNeighborSampler(GraphModelStorage *storage,
                                               std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs,
                                               bool incoming,
                                               bool outgoing) {
    storage_ = storage;
    sampling_layers_ = layer_configs;
    incoming_ = incoming;
    outgoing_ = outgoing;
    use_hashmap_sets_ = false;
    use_bitmaps_ = false;

    for (int i = 0; i < sampling_layers_.size(); i++) {
        if (use_bitmaps_ && sampling_layers_[i]->use_hashmap_sets) {
            throw std::runtime_error("Layers with hashmap sets equal to true must come before those set to false.");
        }
        if (sampling_layers_[i]->use_hashmap_sets) {
            use_hashmap_sets_ = true;
        } else {
            use_bitmaps_ = true;
        }
    }
}

GNNGraph LayeredNeighborSampler::getNeighbors(torch::Tensor node_ids, int worker_id) {
    Indices hop_offsets;
    torch::Tensor incoming_edges;
    Indices incoming_offsets;
    Indices in_neighbors_mapping;
    torch::Tensor outgoing_edges;
    Indices outgoing_offsets;
    Indices out_neighbors_mapping;

    std::vector<torch::Tensor> incoming_edges_vec;
    std::vector<torch::Tensor> outgoing_edges_vec;

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    hop_offsets = torch::zeros({1}, device_options);
    Indices delta_ids = node_ids;

    int gpu = 0;
    if (node_ids.is_cuda()){
        gpu = 1;
    }

    int64_t num_nodes_in_memory = storage_->current_subgraph_state_->in_memory_subgraph_->num_nodes_in_memory_;

    // data structures for calculating the delta_ids
    torch::Tensor hash_map;
//    void *hash_map_mem;
    auto bool_device_options = torch::TensorOptions().dtype(torch::kBool).device(node_ids.device());

    phmap::flat_hash_set<int64_t> seen_unique_nodes;
    phmap::flat_hash_set<int64_t>::const_iterator found;
    vector<int64_t> delta_ids_vec;

    if (gpu) {
        hash_map = torch::zeros({num_nodes_in_memory}, bool_device_options);
    } else {
        if (use_bitmaps_) {
            hash_map = storage_->current_subgraph_state_->in_memory_subgraph_->hash_maps_[worker_id];
        }
        if (use_hashmap_sets_) {
            seen_unique_nodes.reserve(node_ids.size(0));
        }
    }

    for (int i = 0; i < sampling_layers_.size(); i++) {
        torch::Tensor delta_incoming_edges;
        Indices delta_incoming_offsets;
        torch::Tensor delta_outgoing_edges;
        Indices delta_outgoing_offsets;

        NeighborSamplingLayer layer_type = sampling_layers_[i]->type;
        auto options = sampling_layers_[i]->options;

        int max_neighbors = -1;
        float rate = 0.0;
        if (layer_type == NeighborSamplingLayer::UNIFORM) {
            max_neighbors = std::dynamic_pointer_cast<UniformSamplingOptions>(options)->max_neighbors;
        } else if (layer_type == NeighborSamplingLayer::DROPOUT) {
            rate = std::dynamic_pointer_cast<DropoutSamplingOptions>(options)->rate;
        }

        if (delta_ids.size(0) > 0) {
            if (incoming_) {
                auto tup = storage_->current_subgraph_state_->in_memory_subgraph_->getNeighborsForNodeIds(delta_ids, true, layer_type, max_neighbors, rate);
                delta_incoming_edges = std::get<0>(tup);
                delta_incoming_offsets = std::get<1>(tup);
            }

            if (outgoing_) {
                auto tup = storage_->current_subgraph_state_->in_memory_subgraph_->getNeighborsForNodeIds(delta_ids, false, layer_type, max_neighbors, rate);
                delta_outgoing_edges = std::get<0>(tup);
                delta_outgoing_offsets = std::get<1>(tup);
            }
        }

        if (incoming_offsets.defined()) {
            if (delta_incoming_offsets.size(0) > 0) {
                incoming_offsets = incoming_offsets + delta_incoming_edges.size(0);
                incoming_offsets = torch::cat({delta_incoming_offsets, incoming_offsets}, 0);
            }
        } else {
            incoming_offsets = delta_incoming_offsets;
        }
        if (delta_incoming_edges.size(0) > 0) {
            incoming_edges_vec.emplace(incoming_edges_vec.begin(), delta_incoming_edges);
        }

        if (outgoing_offsets.defined()) {
            if (delta_outgoing_edges.size(0) > 0) {
                outgoing_offsets = outgoing_offsets + delta_outgoing_edges.size(0);
                outgoing_offsets = torch::cat({delta_outgoing_offsets, outgoing_offsets}, 0);
            }
        } else {
            outgoing_offsets = delta_outgoing_offsets;
        }
        if (delta_outgoing_edges.size(0) > 0) {
            outgoing_edges_vec.emplace(outgoing_edges_vec.begin(), delta_outgoing_edges);
        }

        // calculate delta_ids
        if (node_ids.device().is_cuda()) {
            if (i > 0) {
                hash_map = 0 * hash_map;
            }

            if (delta_incoming_edges.size(0) > 0) {
                hash_map.index_fill_(0, delta_incoming_edges.select(1, 0), 1);
            }
            if (delta_outgoing_edges.size(0) > 0) {
                hash_map.index_fill_(0, delta_outgoing_edges.select(1, -1), 1);
            }
            hash_map.index_fill_(0, node_ids, 0);

            delta_ids = hash_map.nonzero().flatten(0, 1);
        } else {

            if (!sampling_layers_[i]->use_hashmap_sets) {
                delta_ids = computeDeltaIdsHelperMethod1(hash_map, node_ids, delta_incoming_edges, delta_outgoing_edges,
                                                         num_nodes_in_memory);
            } else {
                delta_ids_vec.clear();

                if (i == 0) {
                    auto nodes_accessor = node_ids.accessor<int64_t , 1>();
                    for (int j = 0; j < node_ids.size(0); j++) {
                        seen_unique_nodes.emplace(nodes_accessor[j]);
                    }
                }

                if (delta_incoming_edges.size(0) > 0) {
                    auto incoming_accessor = delta_incoming_edges.accessor<int64_t , 2>();
                    for (int j = 0; j < delta_incoming_edges.size(0); j++) {
                        found = seen_unique_nodes.find(incoming_accessor[j][0]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(incoming_accessor[j][0]);
                            seen_unique_nodes.emplace(incoming_accessor[j][0]);
                        }
                    }
                }

                if (delta_outgoing_edges.size(0) > 0) {
                    int column_idx = delta_outgoing_edges.size(1) - 1; // RW: -1 has some weird bug for accessor
                    auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t , 2>();
                    for (int j = 0; j < delta_outgoing_edges.size(0); j++) {
                        found = seen_unique_nodes.find(outgoing_accessor[j][column_idx]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(outgoing_accessor[j][column_idx]);
                            seen_unique_nodes.emplace(outgoing_accessor[j][column_idx]);
                        }
                    }
                }

                delta_ids = torch::from_blob(delta_ids_vec.data(), {(int) delta_ids_vec.size()}, torch::kInt64);
            }

        }

        hop_offsets = hop_offsets + delta_ids.size(0);
        hop_offsets = torch::cat({torch::zeros({1}, device_options), hop_offsets});

        if (delta_ids.size(0) > 0) {
            node_ids = torch::cat({delta_ids, node_ids}, 0);
        }
    }
    hop_offsets = torch::cat({hop_offsets, torch::tensor({node_ids.size(0)}, device_options)});

    GNNGraph ret = GNNGraph(hop_offsets, node_ids, incoming_offsets, incoming_edges_vec, in_neighbors_mapping, outgoing_offsets, outgoing_edges_vec, out_neighbors_mapping, num_nodes_in_memory);

    return ret;
}

torch::Tensor LayeredNeighborSampler::computeDeltaIdsHelperMethod1(torch::Tensor hash_map, torch::Tensor node_ids,
                                                                   torch::Tensor delta_incoming_edges,
                                                                   torch::Tensor delta_outgoing_edges,
                                                                   int64_t num_nodes_in_memory) {
    auto hash_map_accessor = hash_map.accessor<bool, 1>();
    auto nodes_accessor = node_ids.accessor<int64_t , 1>();

    if (delta_incoming_edges.size(0) > 0) {
        auto incoming_accessor = delta_incoming_edges.accessor<int64_t , 2>();

        #pragma omp parallel for
        for (int64_t j = 0; j < delta_incoming_edges.size(0); j++) {
            hash_map_accessor[incoming_accessor[j][0]] = 1;
        }
    }

    if (delta_outgoing_edges.size(0) > 0) {
        auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t , 2>();
        int column_idx = delta_outgoing_edges.size(1) - 1; // RW: -1 has some weird bug for accessor

        #pragma omp parallel for
        for (int64_t j = 0; j < delta_outgoing_edges.size(0); j++) {
            hash_map_accessor[outgoing_accessor[j][column_idx]] = 1;
        }
    }

    #pragma omp parallel for
    for (int64_t j = 0; j < node_ids.size(0); j++) {
        hash_map_accessor[nodes_accessor[j]] = 0;
    }

    unsigned int num_threads = 1;
    #ifdef MARIUS_OMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    #endif

    std::vector<int> sub_counts = std::vector<int>(num_threads, 0);
    std::vector<int> sub_offsets = std::vector<int>(num_threads, 0);
    int64_t chunk_size = ceil((double) num_nodes_in_memory / num_threads);

    #pragma omp parallel
    {

        #ifdef MARIUS_OMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

        int64_t start = chunk_size * tid;
        int64_t end = start + chunk_size;

        if (end > num_nodes_in_memory) {
            end = num_nodes_in_memory;
        }

        int private_count = 0;
        for (int64_t j = start; j < end; j++) {
            if (hash_map_accessor[j]) {
                private_count++;
            }
        }
        sub_counts[tid] = private_count;
    }

    int count = 0;
    for (auto c : sub_counts) {
        count += c;
    }

    for (int k = 0; k < num_threads-1; k++) {
        sub_offsets[k+1] = sub_offsets[k] + sub_counts[k];
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    torch::Tensor delta_ids = torch::empty({count}, device_options);
    auto delta_ids_accessor = delta_ids.accessor<int64_t, 1>();

    #pragma omp parallel
    {
        #ifdef MARIUS_OMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif

        int64_t start = chunk_size * tid;
        int64_t end = start + chunk_size;

        if (end > num_nodes_in_memory) {
            end = num_nodes_in_memory;
        }

        int k = sub_offsets[tid];

        for (int64_t j = start; j < end; j++) {
            if (hash_map_accessor[j]) {
                delta_ids_accessor[k++] = j;
            }
        }
    }

    // zero hash map
    if (delta_incoming_edges.size(0) > 0) {
        auto incoming_accessor = delta_incoming_edges.accessor<int64_t , 2>();

        #pragma omp parallel for
        for (int64_t j = 0; j < delta_incoming_edges.size(0); j++) {
            hash_map_accessor[incoming_accessor[j][0]] = 0;
        }
    }
    if (delta_outgoing_edges.size(0) > 0) {
        auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t , 2>();
        int column_idx = delta_outgoing_edges.size(1) - 1; // RW: -1 has some weird bug for accessor

        #pragma omp parallel for
        for (int64_t j = 0; j < delta_outgoing_edges.size(0); j++) {
            hash_map_accessor[outgoing_accessor[j][column_idx]] = 0;
        }
    }

    return delta_ids;
}