//
// Created by Jason Mohoney on 2/8/22.
//

#include "data/samplers/neighbor.h"

#include <parallel_hashmap/phmap.h>

std::tuple<torch::Tensor, torch::Tensor> sample_all_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                        torch::Tensor num_neighbors) {
    torch::Tensor repeated_starts = global_offsets.repeat_interleave(num_neighbors);
    torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
    torch::Tensor arange = torch::arange(repeated_offsets.size(0), edges.options());
    torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;

    return std::forward_as_tuple(edges.index_select(0, sorted_list_idx), local_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> sample_all_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                        torch::Tensor num_neighbors, int64_t total_neighbors) {
    auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
    auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();
    auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

    int num_columns = edges.size(1);

    auto options = edges.options();
#ifdef MARIUS_CUDA
    options = options.pinned_memory(true);
#endif

    Indices ret_neighbor_id_edges = torch::empty({total_neighbors, num_columns}, options);
    int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

    int64_t *sorted_list_ptr = edges.data_ptr<int64_t>();

    if (num_columns == 3) {
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t local_offset = local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                int count = 0;

                // can this be optimized even further?
                for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count))) = *(sorted_list_ptr + (3 * j));
                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 1) = *(sorted_list_ptr + (3 * j) + 1);
                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 2) = *(sorted_list_ptr + (3 * j) + 2);
                    count++;
                }
            }
        }
    } else {
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t local_offset = local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                int count = 0;

                // can this be optimized even further?
                for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count))) = *(sorted_list_ptr + (2 * j));
                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count)) + 1) = *(sorted_list_ptr + (2 * j) + 1);
                    count++;
                }
            }
        }
    }
    return std::forward_as_tuple(ret_neighbor_id_edges, local_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> sample_uniform_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, int64_t max_neighbors, int64_t max_id) {
    torch::Tensor mask = num_neighbors > max_neighbors;

    torch::Tensor capped_num_neighbors = num_neighbors.masked_fill(mask, max_neighbors);
    local_offsets = capped_num_neighbors.cumsum(0) - capped_num_neighbors;

    torch::Tensor repeated_starts = global_offsets.repeat_interleave(capped_num_neighbors);
    torch::Tensor repeated_offsets = local_offsets.repeat_interleave(capped_num_neighbors);
    torch::Tensor arange = torch::arange(repeated_offsets.size(0), edges.options());
    torch::Tensor ranged_sorted_list_idx = repeated_starts + arange - repeated_offsets;

    torch::Tensor repeated_num_neighbors = num_neighbors.repeat_interleave(capped_num_neighbors);
    torch::Tensor rand_samples = torch::randint(max_id, repeated_offsets.sizes(), edges.options());

    rand_samples.fmod_(repeated_num_neighbors);
    torch::Tensor sampled_sorted_list_idx = repeated_starts + rand_samples;

    mask = mask.repeat_interleave(capped_num_neighbors);
    torch::Tensor sorted_list_idx = torch::where(mask, sampled_sorted_list_idx, ranged_sorted_list_idx);

    return std::forward_as_tuple(edges.index_select(0, sorted_list_idx), local_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> sample_uniform_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, int64_t max_neighbors, int64_t total_neighbors) {
    auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
    auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

    auto capped_num_neighbors = num_neighbors.clone();
    auto capped_num_neighbors_accessor = capped_num_neighbors.accessor<int64_t, 1>();
    int64_t *capped_num_neighbors_mem = capped_num_neighbors.data_ptr<int64_t>();

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < local_offsets.size(0); i++) {
        if (capped_num_neighbors_accessor[i] > max_neighbors) {
            *(capped_num_neighbors_mem + i) = max_neighbors;
        }
    }

    int num_columns = edges.size(1);

    torch::Tensor summed_num_neighbors = capped_num_neighbors.cumsum(0);
    local_offsets = summed_num_neighbors - capped_num_neighbors;
    total_neighbors = summed_num_neighbors[-1].item<int64_t>();

    auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

    auto options = edges.options();
#ifdef MARIUS_CUDA
    options = options.pinned_memory(true);
#endif
    Indices ret_neighbor_id_edges = torch::empty({total_neighbors, num_columns}, options);
    int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

    int64_t *sorted_list_ptr = edges.data_ptr<int64_t>();

    // setup seeds
    unsigned int num_threads = 1;

#ifdef MARIUS_OMP
    #pragma omp parallel
    {
    #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif

    std::vector<unsigned int> tid_seeds(num_threads);

    for (int i = 0; i < num_threads; i++) {
        tid_seeds[i] = rand();
    }

    if (num_columns == 3) {
#pragma omp parallel default(none) shared(tid_seeds, local_offsets_accessor, local_offsets, global_offsets_accessor, global_offsets, num_neighbors_accessor, \
                                              num_neighbors, max_neighbors, sorted_list_ptr, edges, ret_neighbor_id_edges_mem, ret_neighbor_id_edges)
        {
#ifdef MARIUS_OMP
            unsigned int seed = tid_seeds[omp_get_thread_num()];
#else
            unsigned int seed = tid_seeds[0];
#endif

#pragma omp for schedule(runtime)
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t local_offset = local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                if (num_edges > max_neighbors) {
                    int count = 0;
                    int64_t rand_id = 0;
#pragma unroll
                    for (int64_t j = 0; j < max_neighbors; j++) {
                        rand_id = 3 * (global_offset + (rand_r(&seed) % num_edges));

                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count))) = *(sorted_list_ptr + rand_id);
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 1) = *(sorted_list_ptr + rand_id + 1);
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 2) = *(sorted_list_ptr + rand_id + 2);
                        count++;
                    }
                } else {
                    int count = 0;
#pragma unroll
                    for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count))) = *(sorted_list_ptr + (3 * j));
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 1) = *(sorted_list_ptr + (3 * j) + 1);
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 2) = *(sorted_list_ptr + (3 * j) + 2);
                        count++;
                    }
                }
            }
        }
    } else {
#pragma omp parallel default(none) shared(tid_seeds, local_offsets_accessor, local_offsets, global_offsets_accessor, global_offsets, num_neighbors_accessor, \
                                              num_neighbors, max_neighbors, sorted_list_ptr, edges, ret_neighbor_id_edges_mem, ret_neighbor_id_edges)
        {
#ifdef MARIUS_OMP
            unsigned int seed = tid_seeds[omp_get_thread_num()];
#else
            unsigned int seed = tid_seeds[0];
#endif

#pragma omp for schedule(runtime)
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t local_offset = local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                if (num_edges > max_neighbors) {
                    int count = 0;
                    int64_t rand_id = 0;
#pragma unroll
                    for (int64_t j = 0; j < max_neighbors; j++) {
                        rand_id = 2 * (global_offset + (rand_r(&seed) % num_edges));

                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + count))) = *(sorted_list_ptr + rand_id);
                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + count)) + 1) = *(sorted_list_ptr + rand_id + 1);
                        count++;
                    }
                } else {
                    int count = 0;
#pragma unroll
                    for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + count))) = *(sorted_list_ptr + (2 * j));
                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + count)) + 1) = *(sorted_list_ptr + (2 * j) + 1);
                        count++;
                    }
                }
            }
        }
    }
    return std::forward_as_tuple(ret_neighbor_id_edges, local_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> sample_dropout_gpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, float rate) {
    torch::Tensor repeated_starts = global_offsets.repeat_interleave(num_neighbors);
    torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
    torch::Tensor arange = torch::arange(repeated_offsets.size(0), edges.options());
    torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;

    torch::Tensor keep_mask = torch::rand(sorted_list_idx.size(0), torch::TensorOptions().device(edges.device()));
    keep_mask = torch::ge(keep_mask, rate);
    sorted_list_idx = sorted_list_idx.masked_select(keep_mask);

    torch::Tensor capped_num_neighbors = segmented_sum_with_offsets(keep_mask.to(torch::kInt64), local_offsets);

    torch::Tensor summed_num_neighbors = capped_num_neighbors.cumsum(0);
    local_offsets = summed_num_neighbors - capped_num_neighbors;

    return std::forward_as_tuple(edges.index_select(0, sorted_list_idx), local_offsets);
}

std::tuple<torch::Tensor, torch::Tensor> sample_dropout_cpu(torch::Tensor edges, torch::Tensor global_offsets, torch::Tensor local_offsets,
                                                            torch::Tensor num_neighbors, float rate, int64_t total_neighbors) {
    auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
    auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();
    auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

    auto capped_num_neighbors = num_neighbors.clone();
    int64_t *capped_num_neighbors_mem = capped_num_neighbors.data_ptr<int64_t>();

    torch::Tensor keep_mask = torch::rand(total_neighbors, edges.device());
    auto keep_mask_accessor = keep_mask.accessor<float, 1>();

    int num_columns = edges.size(1);

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < local_offsets.size(0); i++) {
            int64_t local_offset = local_offsets_accessor[i];
            int64_t num_edges = num_neighbors_accessor[i];

            int count = 0;
            for (int j = local_offset; j < local_offset + num_edges; j++) {
                if (keep_mask_accessor[j] >= rate) {
                    count++;
                }
            }
            *(capped_num_neighbors_mem + i) = count;
        }
    }

    torch::Tensor summed_num_neighbors = capped_num_neighbors.cumsum(0);
    Indices new_local_offsets = summed_num_neighbors - capped_num_neighbors;
    total_neighbors = summed_num_neighbors[-1].item<int64_t>();

    auto new_local_offsets_accessor = new_local_offsets.accessor<int64_t, 1>();

    auto options = edges.options();
#ifdef MARIUS_CUDA
    options = options.pinned_memory(true);
#endif
    Indices ret_neighbor_id_edges = torch::empty({total_neighbors, 3}, options);
    int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

    int64_t *sorted_list_ptr = edges.data_ptr<int64_t>();

    if (num_columns == 3) {
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t old_local_offset = local_offsets_accessor[i];
                int64_t local_offset = new_local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                int local_count = 0;
                int global_count = 0;

                // can this be optimized even further?
                for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                    if (keep_mask_accessor[old_local_offset + global_count] >= rate) {
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + local_count))) = *(sorted_list_ptr + (3 * j));
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + local_count)) + 1) = *(sorted_list_ptr + (3 * j) + 1);
                        *(ret_neighbor_id_edges_mem + (3 * (local_offset + local_count)) + 2) = *(sorted_list_ptr + (3 * j) + 2);
                        local_count++;
                    }
                    global_count++;
                }
            }
        }
    } else {
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < local_offsets.size(0); i++) {
                int64_t old_local_offset = local_offsets_accessor[i];
                int64_t local_offset = new_local_offsets_accessor[i];
                int64_t global_offset = global_offsets_accessor[i];
                int64_t num_edges = num_neighbors_accessor[i];

                int local_count = 0;
                int global_count = 0;

                // can this be optimized even further?
                for (int64_t j = global_offset; j < global_offset + num_edges; j++) {
                    if (keep_mask_accessor[old_local_offset + global_count] >= rate) {
                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + local_count))) = *(sorted_list_ptr + (2 * j));
                        *(ret_neighbor_id_edges_mem + (2 * (local_offset + local_count)) + 1) = *(sorted_list_ptr + (2 * j) + 1);
                        local_count++;
                    }
                    global_count++;
                }
            }
        }
    }
    return std::forward_as_tuple(ret_neighbor_id_edges, new_local_offsets);
}

LayeredNeighborSampler::LayeredNeighborSampler(shared_ptr<GraphModelStorage> storage, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs,
                                               bool use_incoming_nbrs, bool use_outgoing_nbrs) {
    storage_ = storage;
    graph_ = nullptr;
    sampling_layers_ = layer_configs;
    use_incoming_nbrs_ = use_incoming_nbrs;
    use_outgoing_nbrs_ = use_outgoing_nbrs;

    checkLayerConfigs();
}

LayeredNeighborSampler::LayeredNeighborSampler(shared_ptr<MariusGraph> graph, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, torch::Tensor in_mem_nodes, 
                                               shared_ptr<FeaturesLoaderConfig> features_config, bool use_incoming_nbrs, bool use_outgoing_nbrs) {
    
    graph_ = graph;
    storage_ = nullptr;
    sampling_layers_ = layer_configs;
    use_incoming_nbrs_ = use_incoming_nbrs;
    use_outgoing_nbrs_ = use_outgoing_nbrs;
    in_mem_nodes_ = in_mem_nodes;
    features_loader_ = get_feature_loader(features_config, graph);

    checkLayerConfigs();

}

LayeredNeighborSampler::LayeredNeighborSampler(shared_ptr<MariusGraph> graph, std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs,
                                               bool use_incoming_nbrs, bool use_outgoing_nbrs) {
    graph_ = graph;
    storage_ = nullptr;
    sampling_layers_ = layer_configs;
    use_incoming_nbrs_ = use_incoming_nbrs;
    use_outgoing_nbrs_ = use_outgoing_nbrs;

    checkLayerConfigs();
}

LayeredNeighborSampler::LayeredNeighborSampler(std::vector<shared_ptr<NeighborSamplingConfig>> layer_configs, bool use_incoming_nbrs, bool use_outgoing_nbrs) {
    graph_ = nullptr;
    storage_ = nullptr;
    sampling_layers_ = layer_configs;
    use_incoming_nbrs_ = use_incoming_nbrs;
    use_outgoing_nbrs_ = use_outgoing_nbrs;

    checkLayerConfigs();
}

void LayeredNeighborSampler::checkLayerConfigs() {
    use_hashmap_sets_ = false;
    use_bitmaps_ = false;

    for (int i = 0; i < sampling_layers_.size(); i++) {
        if (use_bitmaps_ && sampling_layers_[i]->use_hashmap_sets) {
            throw std::runtime_error("Layers with use_hashmap_sets equal to true must come before those set to false.");
        }
        if (sampling_layers_[i]->use_hashmap_sets) {
            use_hashmap_sets_ = true;
        } else {
            use_bitmaps_ = true;
        }
    }
}

DENSEGraph LayeredNeighborSampler::getNeighbors(torch::Tensor node_ids, shared_ptr<MariusGraph> graph, int worker_id) {
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
    if (node_ids.is_cuda()) {
        gpu = 1;
    }

    if (graph == nullptr) {
        if (storage_ != nullptr) {
            graph = storage_->current_subgraph_state_->in_memory_subgraph_;
        } else if (graph_ != nullptr) {
            graph = graph_;
        } else {
            throw MariusRuntimeException("Graph to sample from is undefined");
        }
    }

    int64_t num_nodes_in_memory = graph->num_nodes_in_memory_;

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
            hash_map = graph->hash_maps_[worker_id];
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
            if (use_incoming_nbrs_) {
                auto tup = graph->getNeighborsForNodeIds(delta_ids, true, layer_type, max_neighbors, rate);
                delta_incoming_edges = std::get<0>(tup);
                delta_incoming_offsets = std::get<1>(tup);
            }

            if (use_outgoing_nbrs_) {
                auto tup = graph->getNeighborsForNodeIds(delta_ids, false, layer_type, max_neighbors, rate);
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
            if (delta_outgoing_offsets.size(0) > 0) {
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
                delta_ids = computeDeltaIdsHelperMethod1(hash_map, node_ids, delta_incoming_edges, delta_outgoing_edges, num_nodes_in_memory);
            } else {
                delta_ids_vec.clear();

                if (i == 0) {
                    auto nodes_accessor = node_ids.accessor<int64_t, 1>();
                    for (int j = 0; j < node_ids.size(0); j++) {
                        seen_unique_nodes.emplace(nodes_accessor[j]);
                    }
                }

                if (delta_incoming_edges.size(0) > 0) {
                    auto incoming_accessor = delta_incoming_edges.accessor<int64_t, 2>();
                    for (int j = 0; j < delta_incoming_edges.size(0); j++) {
                        found = seen_unique_nodes.find(incoming_accessor[j][0]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(incoming_accessor[j][0]);
                            seen_unique_nodes.emplace(incoming_accessor[j][0]);
                        }
                    }
                }

                if (delta_outgoing_edges.size(0) > 0) {
                    int column_idx = delta_outgoing_edges.size(1) - 1;  // RW: -1 has some weird bug for accessor
                    auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t, 2>();
                    for (int j = 0; j < delta_outgoing_edges.size(0); j++) {
                        found = seen_unique_nodes.find(outgoing_accessor[j][column_idx]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(outgoing_accessor[j][column_idx]);
                            seen_unique_nodes.emplace(outgoing_accessor[j][column_idx]);
                        }
                    }
                }

                delta_ids = torch::from_blob(delta_ids_vec.data(), {(int)delta_ids_vec.size()}, torch::kInt64);
            }
        }

        std::cout << "Level " << i << " has " << delta_ids.numel() << " nodes" << std::endl; 
        hop_offsets = hop_offsets + delta_ids.size(0);
        hop_offsets = torch::cat({torch::zeros({1}, device_options), hop_offsets});

        if (delta_ids.size(0) > 0) {
            node_ids = torch::cat({delta_ids, node_ids}, 0);
        }
    }
    hop_offsets = torch::cat({hop_offsets, torch::tensor({node_ids.size(0)}, device_options)});

    DENSEGraph ret = DENSEGraph(hop_offsets, node_ids, incoming_offsets, incoming_edges_vec, in_neighbors_mapping, outgoing_offsets, outgoing_edges_vec,
                                out_neighbors_mapping, num_nodes_in_memory);

    //    if (!gpu and use_bitmaps_) {
    //        free(hash_map_mem);
    //    }

    return ret;
}

float LayeredNeighborSampler::getAvgPercentRemoved() {
    if(percent_count_ == 0) { 
        return -1.0;
    }
    return  percent_removed_total_/percent_count_;
}

float LayeredNeighborSampler::getAvgScalingFactor() {
    if(scaling_count_ == 0) { 
        return -1.0;
    }
    return scaling_factor_total_/scaling_count_;
}

torch::Tensor LayeredNeighborSampler::remove_in_mem_nodes(torch::Tensor node_ids) {
    if(in_mem_nodes_.defined()) {
        int64_t initial_node_count = node_ids.numel();
        node_ids = torch::masked_select(node_ids, ~torch::isin(node_ids, in_mem_nodes_));
        int64_t post_remove_count = node_ids.numel();
        float percent_removed = (100.0 * (initial_node_count - post_remove_count))/initial_node_count;

        // Update the percent
        percent_removed_total_ += percent_removed;
        percent_count_ += 1;
    }

    return node_ids;
}

int64_t LayeredNeighborSampler::getNeighborsPages(torch::Tensor node_ids, shared_ptr<MariusGraph> graph, int worker_id) {
    torch::Tensor incoming_edges;
    Indices incoming_offsets;
    Indices in_neighbors_mapping;
    torch::Tensor outgoing_edges;
    Indices outgoing_offsets;
    Indices out_neighbors_mapping;

    std::vector<torch::Tensor> incoming_edges_vec;
    std::vector<torch::Tensor> outgoing_edges_vec;

    node_ids = remove_in_mem_nodes(node_ids);
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    Indices delta_ids = node_ids;

    int gpu = 0;
    if (node_ids.is_cuda()) {
        gpu = 1;
    }

    if (graph == nullptr) {
        if (storage_ != nullptr) {
            graph = storage_->current_subgraph_state_->in_memory_subgraph_;
        } else if (graph_ != nullptr) {
            graph = graph_;
        } else {
            throw MariusRuntimeException("Graph to sample from is undefined");
        }
    }

    int64_t num_nodes_in_memory = graph->num_nodes_in_memory_;

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
            hash_map = graph->hash_maps_[worker_id];
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

        int64_t initial_nodes = delta_ids.numel();
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
            if (use_incoming_nbrs_) {
                auto tup = graph->getNeighborsForNodeIds(delta_ids, true, layer_type, max_neighbors, rate);
                delta_incoming_edges = std::get<0>(tup);
                delta_incoming_offsets = std::get<1>(tup);
            }

            if (use_outgoing_nbrs_) {
                auto tup = graph->getNeighborsForNodeIds(delta_ids, false, layer_type, max_neighbors, rate);
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
            if (delta_outgoing_offsets.size(0) > 0) {
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
                delta_ids = computeDeltaIdsHelperMethod1(hash_map, node_ids, delta_incoming_edges, delta_outgoing_edges, num_nodes_in_memory);
            } else {
                delta_ids_vec.clear();

                if (i == 0) {
                    auto nodes_accessor = node_ids.accessor<int64_t, 1>();
                    for (int j = 0; j < node_ids.size(0); j++) {
                        seen_unique_nodes.emplace(nodes_accessor[j]);
                    }
                }

                if (delta_incoming_edges.size(0) > 0) {
                    auto incoming_accessor = delta_incoming_edges.accessor<int64_t, 2>();
                    for (int j = 0; j < delta_incoming_edges.size(0); j++) {
                        found = seen_unique_nodes.find(incoming_accessor[j][0]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(incoming_accessor[j][0]);
                            seen_unique_nodes.emplace(incoming_accessor[j][0]);
                        }
                    }
                }

                if (delta_outgoing_edges.size(0) > 0) {
                    int column_idx = delta_outgoing_edges.size(1) - 1;  // RW: -1 has some weird bug for accessor
                    auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t, 2>();
                    for (int j = 0; j < delta_outgoing_edges.size(0); j++) {
                        found = seen_unique_nodes.find(outgoing_accessor[j][column_idx]);
                        if (found == seen_unique_nodes.end()) {
                            delta_ids_vec.emplace_back(outgoing_accessor[j][column_idx]);
                            seen_unique_nodes.emplace(outgoing_accessor[j][column_idx]);
                        }
                    }
                }

                delta_ids = torch::from_blob(delta_ids_vec.data(), {(int)delta_ids_vec.size()}, torch::kInt64);
            }
        }

        delta_ids = remove_in_mem_nodes(delta_ids);

        // Calculate the scaling factor
        int64_t next_nodes = delta_ids.numel();
        float scaling_factor = (1.0 * next_nodes)/initial_nodes;
        scaling_factor_total_ += scaling_factor;
        scaling_count_ += 1;

        if (delta_ids.size(0) > 0) {
            node_ids = torch::cat({delta_ids, node_ids}, 0);
        }
    }

    return features_loader_->num_pages_for_nodes(node_ids);
}

torch::Tensor LayeredNeighborSampler::computeDeltaIdsHelperMethod1(torch::Tensor hash_map, torch::Tensor node_ids, torch::Tensor delta_incoming_edges,
                                                                   torch::Tensor delta_outgoing_edges, int64_t num_nodes_in_memory) {
    unsigned int num_threads = 1;
#ifdef MARIUS_OMP
    #pragma omp parallel
    {
    #pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif

    int64_t chunk_size = ceil((double)num_nodes_in_memory / num_threads);

    auto hash_map_accessor = hash_map.accessor<bool, 1>();
    auto nodes_accessor = node_ids.accessor<int64_t, 1>();

#pragma omp parallel default(none) shared(delta_incoming_edges, delta_outgoing_edges, hash_map_accessor, hash_map, node_ids, nodes_accessor)
    {
        if (delta_incoming_edges.size(0) > 0) {
            auto incoming_accessor = delta_incoming_edges.accessor<int64_t, 2>();

#pragma omp for  // nowait -> can't have this because of the below if statement skipping directly to node ids for loop
            for (int64_t j = 0; j < delta_incoming_edges.size(0); j++) {
                if (!hash_map_accessor[incoming_accessor[j][0]]) {
                    hash_map_accessor[incoming_accessor[j][0]] = 1;
                }
            }
        }

        if (delta_outgoing_edges.size(0) > 0) {
            auto outgoing_accessor = delta_outgoing_edges.accessor<int64_t, 2>();
            int column_idx = delta_outgoing_edges.size(1) - 1;  // RW: -1 has some weird bug for accessor

#pragma omp for
            for (int64_t j = 0; j < delta_outgoing_edges.size(0); j++) {
                if (!hash_map_accessor[outgoing_accessor[j][column_idx]]) {
                    hash_map_accessor[outgoing_accessor[j][column_idx]] = 1;
                }
            }
        }

#pragma omp for
        for (int64_t j = 0; j < node_ids.size(0); j++) {
            if (hash_map_accessor[nodes_accessor[j]]) {
                hash_map_accessor[nodes_accessor[j]] = 0;
            }
        }
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    std::vector<torch::Tensor> sub_deltas = std::vector<torch::Tensor>(num_threads);
    int64_t upper_bound = (int64_t)(delta_incoming_edges.size(0) + delta_outgoing_edges.size(0)) / num_threads + 1;

    std::vector<int> sub_counts = std::vector<int>(num_threads, 0);
    std::vector<int> sub_offsets = std::vector<int>(num_threads, 0);

#pragma omp parallel
    {
#ifdef MARIUS_OMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        sub_deltas[tid] = torch::empty({upper_bound}, device_options);
        auto delta_ids_accessor = sub_deltas[tid].accessor<int64_t, 1>();

        int64_t start = chunk_size * tid;
        int64_t end = start + chunk_size;

        if (end > num_nodes_in_memory) {
            end = num_nodes_in_memory;
        }

        int private_count = 0;
        int grow_count = 0;

#pragma unroll
        for (int64_t j = start; j < end; j++) {
            if (hash_map_accessor[j]) {
                delta_ids_accessor[private_count++] = j;
                hash_map_accessor[j] = 0;
                grow_count++;

                if (grow_count == upper_bound) {
                    sub_deltas[tid] = torch::cat({sub_deltas[tid], torch::empty({upper_bound}, device_options)}, 0);
                    delta_ids_accessor = sub_deltas[tid].accessor<int64_t, 1>();
                    grow_count = 0;
                }
            }
        }
        sub_counts[tid] = private_count;
    }

    int count = 0;
    for (auto c : sub_counts) {
        count += c;
    }

    for (int k = 0; k < num_threads - 1; k++) {
        sub_offsets[k + 1] = sub_offsets[k] + sub_counts[k];
    }

    torch::Tensor delta_ids = torch::empty({count}, device_options);

#pragma omp parallel for
    for (int k = 0; k < num_threads; k++) {
        delta_ids.narrow(0, sub_offsets[k], sub_counts[k]) = sub_deltas[k].narrow(0, 0, sub_counts[k]);
    }

    return delta_ids;
}