//
// Created by Jason Mohoney on 8/25/21.
//

#include "graph.h"
#include "util.h"

#ifdef MARIUS_OMP
#include "omp.h"
#endif

MariusGraph::MariusGraph() {};

MariusGraph::MariusGraph(EdgeList src_sorted_edges, EdgeList dst_sorted_edges, int64_t num_nodes_in_memory, int num_hash_maps) {

    num_nodes_in_memory_ = num_nodes_in_memory;

    src_sorted_edges_ = src_sorted_edges;
    dst_sorted_edges_ = dst_sorted_edges;

    auto contiguous_src = src_sorted_edges_.select(1, 0).contiguous();
    auto contiguous_dst = dst_sorted_edges_.select(1, -1).contiguous();
    torch::Tensor arange_tensor = torch::arange(0, num_nodes_in_memory_, contiguous_src.device());

    out_offsets_ = torch::searchsorted(contiguous_src, arange_tensor);
    torch::Tensor end = torch::tensor({contiguous_src.size(0)}, contiguous_src.options());
    out_num_neighbors_ = torch::cat({out_offsets_, end}).narrow(0, 1, out_offsets_.size(0)) - out_offsets_;

    in_offsets_ = torch::searchsorted(contiguous_dst, arange_tensor);
    end = torch::tensor({contiguous_dst.size(0)}, contiguous_dst.options());
    in_num_neighbors_ = torch::cat({in_offsets_, end}).narrow(0, 1, in_offsets_.size(0)) - in_offsets_;

    max_out_num_neighbors_ = torch::max(out_num_neighbors_).item<int>();
    max_in_num_neighbors_ = torch::max(in_num_neighbors_).item<int>();

    num_hash_maps_ = num_hash_maps;
    if (num_hash_maps_ > 0) {
        auto bool_device_options = torch::TensorOptions().dtype(torch::kBool).device(contiguous_src.device());
        for (int i; i < num_hash_maps_; i++) {
            hash_maps_.emplace_back(torch::zeros({num_nodes_in_memory}, bool_device_options));
        }
    }
}

MariusGraph::~MariusGraph() {
    clear();
}

Indices MariusGraph::getNodeIDs() {
    return node_ids_;
}

Indices MariusGraph::getEdges(bool incoming) {
    if (incoming) {
        return dst_sorted_edges_;
    } else {
        return src_sorted_edges_;
    }
}

Indices MariusGraph::getRelationIDs(bool incoming) {
    // NOTE: need to update for GNNGraph if we no longer store "self" in the edges
    if (src_sorted_edges_.size(1) == 2) {
        return torch::Tensor();
    } else {
        if (incoming) {
            return dst_sorted_edges_.select(1, 1);
        } else {
            return src_sorted_edges_.select(1, 1);
        }
    }
}

Indices MariusGraph::getNeighborOffsets(bool incoming) {
    if (incoming) {
        return in_offsets_;
    } else {
        return out_offsets_;
    }
}

Indices MariusGraph::getNumNeighbors(bool incoming) {
    if (incoming) {
        return in_num_neighbors_;
    } else {
        return out_num_neighbors_;
    }
}

void MariusGraph::clear() {
    node_ids_ = torch::Tensor();
    src_sorted_edges_ = torch::Tensor();
    dst_sorted_edges_ = torch::Tensor();
    active_in_memory_subgraph_ = torch::Tensor();
    out_sorted_uniques_ = torch::Tensor();
    out_offsets_ = torch::Tensor();
    out_num_neighbors_ = torch::Tensor();
    in_sorted_uniques_ = torch::Tensor();
    in_offsets_ = torch::Tensor();
    in_num_neighbors_ = torch::Tensor();

    for (int i; i < hash_maps_.size(); i++) {
        hash_maps_[i] = torch::Tensor();
    }
    hash_maps_ = {};
}

void MariusGraph::to(torch::Device device) {
    node_ids_ = node_ids_.to(device);
    src_sorted_edges_ = src_sorted_edges_.to(device);
    dst_sorted_edges_ = dst_sorted_edges_.to(device);
    out_sorted_uniques_= out_sorted_uniques_.to(device);
    out_offsets_ = out_offsets_.to(device);
    out_num_neighbors_ = out_num_neighbors_.to(device);
    in_sorted_uniques_ = in_sorted_uniques_.to(device);
    in_offsets_ = in_offsets_.to(device);
}

// 1 hop sampler
std::tuple<torch::Tensor, torch::Tensor> MariusGraph::getNeighborsForNodeIds(torch::Tensor node_ids, bool incoming, NeighborSamplingLayer neighbor_sampling_layer, int max_neighbors_size, float rate) {
    int gpu = 0;

    if (node_ids.is_cuda()) {
        gpu = 1;
    }

    Indices in_memory_ids;
    torch::Tensor mask;
    torch::Tensor num_neighbors = torch::empty_like(node_ids);
    Indices global_offsets = torch::empty_like(node_ids);

    if (incoming) {
        if (gpu) {
            num_neighbors = in_num_neighbors_.index_select(0, node_ids);
            global_offsets = in_offsets_.index_select(0, node_ids);
        } else {
            auto in_num_neighbors_accessor = in_num_neighbors_.accessor<int64_t, 1>();
            auto in_offsets_accessor = in_offsets_.accessor<int64_t, 1>();

            auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();
            auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
            auto node_ids_accessor = node_ids.accessor<int64_t, 1>();

            #pragma omp parallel for
            for (int64_t i = 0; i < node_ids.size(0); i++) {
                num_neighbors_accessor[i] = in_num_neighbors_accessor[node_ids_accessor[i]];
                global_offsets_accessor[i] = in_offsets_accessor[node_ids_accessor[i]];
            }
        }
    } else {
        if (gpu) {
            num_neighbors = out_num_neighbors_.index_select(0, node_ids);
            global_offsets = out_offsets_.index_select(0, node_ids);
        } else {
            auto out_num_neighbors_accessor = out_num_neighbors_.accessor<int64_t, 1>();
            auto out_offsets_accessor = out_offsets_.accessor<int64_t, 1>();

            auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();
            auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
            auto node_ids_accessor = node_ids.accessor<int64_t, 1>();

            #pragma omp parallel for
            for (int64_t i = 0; i < node_ids.size(0); i++) {
                num_neighbors_accessor[i] = out_num_neighbors_accessor[node_ids_accessor[i]];
                global_offsets_accessor[i] = out_offsets_accessor[node_ids_accessor[i]];
            }
        }
    }

    int num_columns = src_sorted_edges_.size(1);

    torch::Tensor summed_num_neighbors;
    Indices local_offsets;
    int64_t total_neighbors;
    if (neighbor_sampling_layer != NeighborSamplingLayer::UNIFORM or gpu) {
        summed_num_neighbors = num_neighbors.cumsum(0);
        local_offsets = summed_num_neighbors - num_neighbors;
        total_neighbors = summed_num_neighbors[-1].item<int64_t>();
    }

    std::tuple<torch::Tensor, torch::Tensor> ret;

    // TODO break up this case switch into helper functions for GPU/CPU sampling for each case. Will improve readability and testing ability
    switch (neighbor_sampling_layer) {
        case NeighborSamplingLayer::ALL: {

            if (gpu) {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());

                torch::Tensor repeated_starts = global_offsets.repeat_interleave(num_neighbors);
                torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
                torch::Tensor arange = torch::arange(repeated_offsets.size(0), device_options);
                torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;

                if (incoming) {
                    ret = std::forward_as_tuple(dst_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                } else {
                    ret = std::forward_as_tuple(src_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                }

            } else {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device()).pinned_memory(true);

                auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
                auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();
                auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

                Indices ret_neighbor_id_edges = torch::empty({total_neighbors, num_columns}, device_options);
                int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

                int64_t *sorted_list_ptr;

                if (incoming) {
                    sorted_list_ptr = dst_sorted_edges_.data_ptr<int64_t>();
                } else {
                    sorted_list_ptr = src_sorted_edges_.data_ptr<int64_t>();
                }

                if (num_columns == 3) {
                    #pragma omp parallel
                    {
                        #pragma omp for
                        for (int i = 0; i < node_ids.size(0); i++) {
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
                        for (int i = 0; i < node_ids.size(0); i++) {
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
                ret = std::forward_as_tuple(ret_neighbor_id_edges, local_offsets);
            }

            break;
        }
        case NeighborSamplingLayer::UNIFORM: {

            if (gpu) {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());

                torch::Tensor mask = num_neighbors > max_neighbors_size;

                torch::Tensor capped_num_neighbors = num_neighbors.masked_fill(mask, max_neighbors_size);
                local_offsets = capped_num_neighbors.cumsum(0) - capped_num_neighbors;

                torch::Tensor repeated_starts = global_offsets.repeat_interleave(capped_num_neighbors);
                torch::Tensor repeated_offsets = local_offsets.repeat_interleave(capped_num_neighbors);
                torch::Tensor arange = torch::arange(repeated_offsets.size(0), device_options);
                torch::Tensor ranged_sorted_list_idx = repeated_starts + arange - repeated_offsets;

                torch::Tensor repeated_num_neighbors = num_neighbors.repeat_interleave(capped_num_neighbors);
                torch::Tensor rand_samples;
                if (incoming) {
                    rand_samples = torch::randint(max_in_num_neighbors_, repeated_offsets.sizes(), device_options);
                }
                else {
                    rand_samples = torch::randint(max_out_num_neighbors_, repeated_offsets.sizes(), device_options);
                }
                rand_samples.fmod_(repeated_num_neighbors);
                torch::Tensor sampled_sorted_list_idx = repeated_starts + rand_samples;

                mask = mask.repeat_interleave(capped_num_neighbors);
                torch::Tensor sorted_list_idx = torch::where(mask, sampled_sorted_list_idx, ranged_sorted_list_idx);

                if (incoming) {
                    ret = std::forward_as_tuple(dst_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                } else {
                    ret = std::forward_as_tuple(src_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                }

            } else {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device()).pinned_memory(true);

                auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
                auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

                auto capped_num_neighbors = num_neighbors.clone();
                auto capped_num_neighbors_accessor = capped_num_neighbors.accessor<int64_t, 1>();
                int64_t *capped_num_neighbors_mem = capped_num_neighbors.data_ptr<int64_t>();

                #pragma omp parallel for schedule(runtime)
                for (int i = 0; i < node_ids.size(0); i++) {
                    if (capped_num_neighbors_accessor[i] > max_neighbors_size) {
                        *(capped_num_neighbors_mem + i) = max_neighbors_size;
                    }
                }

                summed_num_neighbors = capped_num_neighbors.cumsum(0);
                local_offsets = summed_num_neighbors - capped_num_neighbors;
                total_neighbors = summed_num_neighbors[-1].item<int64_t>();

                auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

//                Indices ret_neighbor_id_edges = torch::empty({total_neighbors, num_columns-1}, device_options);
                Indices ret_neighbor_id_edges = torch::empty({total_neighbors, num_columns}, device_options);
                int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

                int64_t *sorted_list_ptr;

                if (incoming) {
                    sorted_list_ptr = dst_sorted_edges_.data_ptr<int64_t>();
                } else {
                    sorted_list_ptr = src_sorted_edges_.data_ptr<int64_t>();
                }

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
                    #pragma omp parallel
                    {
                        #ifdef MARIUS_OMP
                        unsigned int seed = tid_seeds[omp_get_thread_num()];
                        #else
                        unsigned int seed = tid_seeds[0];
                        #endif

                        #pragma omp for
                        for (int i = 0; i < node_ids.size(0); i++) {
                            int64_t local_offset = local_offsets_accessor[i];
                            int64_t global_offset = global_offsets_accessor[i];
                            int64_t num_edges = num_neighbors_accessor[i];

                            if (num_edges > max_neighbors_size) {
                                int count = 0;
                                int64_t rand_id = 0;
                                for (int64_t j = 0; j < max_neighbors_size; j++) {

                                    rand_id = 3 * (global_offset + (rand_r(&seed) % num_edges));

                                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count))) = *(sorted_list_ptr + rand_id);
                                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 1) = *(sorted_list_ptr + rand_id + 1);
                                    *(ret_neighbor_id_edges_mem + (3 * (local_offset + count)) + 2) = *(sorted_list_ptr + rand_id + 2);
                                    count++;
                                }
                            } else {
                                int count = 0;
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
                    #pragma omp parallel default(none) \
                        shared(tid_seeds, node_ids, local_offsets_accessor, local_offsets, global_offsets_accessor, global_offsets, \
                        num_neighbors_accessor, num_neighbors, max_neighbors_size, sorted_list_ptr, dst_sorted_edges_, src_sorted_edges_, \
                        ret_neighbor_id_edges_mem, ret_neighbor_id_edges)
                    {
                        #ifdef MARIUS_OMP
                        unsigned int seed = tid_seeds[omp_get_thread_num()];
                        #else
                        unsigned int seed = tid_seeds[0];
                        #endif

//                        int64_t column_offset = 0;
//                        if (!incoming) column_offset = 1;

                        #pragma omp for schedule(runtime)
                        for (int i = 0; i < node_ids.size(0); i++) {
                            int64_t local_offset = local_offsets_accessor[i];
                            int64_t global_offset = global_offsets_accessor[i];
                            int64_t num_edges = num_neighbors_accessor[i];

                            if (num_edges > max_neighbors_size) {
                                int count = 0;
                                int64_t rand_id = 0;
                                #pragma unroll
                                for (int64_t j = 0; j < max_neighbors_size; j++) {

                                    rand_id = 2 * (global_offset + (rand_r(&seed) % num_edges));

//                                    *(ret_neighbor_id_edges_mem + local_offset + count) = *(sorted_list_ptr + rand_id + column_offset);
                                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count))) = *(sorted_list_ptr + rand_id);
                                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count)) + 1) = *(sorted_list_ptr + rand_id + 1);
                                    count++;
                                }
                            } else {
                                int count = 0;
                                #pragma unroll
                                for (int64_t j = global_offset; j < global_offset + num_edges; j++) {

//                                    *(ret_neighbor_id_edges_mem + local_offset + count) = *(sorted_list_ptr + (2 * j) + column_offset);
                                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count))) = *(sorted_list_ptr + (2 * j));
                                    *(ret_neighbor_id_edges_mem + (2 * (local_offset + count)) + 1) = *(sorted_list_ptr + (2 * j) + 1);
                                    count++;
                                }
                            }
                        }
                    }
                }
                ret = std::forward_as_tuple(ret_neighbor_id_edges, local_offsets);
            }

            break;
        }
        case NeighborSamplingLayer::DROPOUT: {

            if (gpu) {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());

                torch::Tensor repeated_starts = global_offsets.repeat_interleave(num_neighbors);
                torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
                torch::Tensor arange = torch::arange(repeated_offsets.size(0), device_options);
                torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;

                torch::Tensor keep_mask = torch::rand(sorted_list_idx.size(0), torch::TensorOptions().device(node_ids.device()));
                keep_mask = torch::ge(keep_mask, rate);
                sorted_list_idx = sorted_list_idx.masked_select(keep_mask);

                torch::Tensor capped_num_neighbors = segmented_sum_with_offsets(keep_mask.to(torch::kInt64), local_offsets);

                summed_num_neighbors = capped_num_neighbors.cumsum(0);
                local_offsets = summed_num_neighbors - capped_num_neighbors;

                if (incoming) {
                    ret = std::forward_as_tuple(dst_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                } else {
                    ret = std::forward_as_tuple(src_sorted_edges_.index_select(0, sorted_list_idx), local_offsets);
                }

            } else {
                auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device()).pinned_memory(true);

                auto global_offsets_accessor = global_offsets.accessor<int64_t, 1>();
                auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();
                auto num_neighbors_accessor = num_neighbors.accessor<int64_t, 1>();

                auto capped_num_neighbors = num_neighbors.clone();
                int64_t *capped_num_neighbors_mem = capped_num_neighbors.data_ptr<int64_t>();

                torch::Tensor keep_mask = torch::rand(total_neighbors, torch::TensorOptions().device(node_ids.device()));
                auto keep_mask_accessor = keep_mask.accessor<float, 1>();

                #pragma omp parallel
                {
                    #pragma omp for
                    for (int i = 0; i < node_ids.size(0); i++) {
                        int64_t local_offset = local_offsets_accessor[i];
                        int64_t num_edges = num_neighbors_accessor[i];

                        int count = 0;
                        for (int j = local_offset; j < local_offset + num_edges; j++) {
                            if (keep_mask_accessor[j] >= rate){
                                count++;
                            }
                        }
                        *(capped_num_neighbors_mem + i) = count;
                    }
                }

                summed_num_neighbors = capped_num_neighbors.cumsum(0);
                Indices new_local_offsets = summed_num_neighbors - capped_num_neighbors;
                total_neighbors = summed_num_neighbors[-1].item<int64_t>();

                auto new_local_offsets_accessor = new_local_offsets.accessor<int64_t, 1>();

                Indices ret_neighbor_id_edges = torch::empty({total_neighbors, 3}, device_options);
                int64_t *ret_neighbor_id_edges_mem = ret_neighbor_id_edges.data_ptr<int64_t>();

                int64_t *sorted_list_ptr;

                if (incoming) {
                    sorted_list_ptr = dst_sorted_edges_.data_ptr<int64_t>();
                } else {
                    sorted_list_ptr = src_sorted_edges_.data_ptr<int64_t>();
                }

                if (num_columns == 3) {
                    #pragma omp parallel
                    {
                        #pragma omp for
                        for (int i = 0; i < node_ids.size(0); i++) {
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
                        for (int i = 0; i < node_ids.size(0); i++) {
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
                ret = std::forward_as_tuple(ret_neighbor_id_edges, new_local_offsets);
            }
            break;
        }
    }
    return ret;
}

GNNGraph::GNNGraph() {};

GNNGraph::GNNGraph(Indices hop_offsets, Indices node_ids, Indices in_offsets, std::vector<torch::Tensor> in_neighbors_vec, Indices in_neighbors_mapping, Indices out_offsets, std::vector<torch::Tensor> out_neighbors_vec, Indices out_neighbors_mapping, int num_nodes_in_memory) {
    hop_offsets_ = hop_offsets;
    node_ids_ = node_ids;
    in_offsets_ = in_offsets;
    in_neighbors_vec_ = in_neighbors_vec;
    in_neighbors_mapping_ = in_neighbors_mapping;
    out_offsets_ = out_offsets;
    out_neighbors_vec_ = out_neighbors_vec;
    out_neighbors_mapping_ = out_neighbors_mapping;
    num_nodes_in_memory_ = num_nodes_in_memory;
}

GNNGraph::~GNNGraph() {
    clear();
}

void GNNGraph::clear() {
    MariusGraph::clear();

    hop_offsets_ = torch::Tensor();

    in_neighbors_mapping_ = torch::Tensor();
    out_neighbors_mapping_ = torch::Tensor();

    in_neighbors_vec_ = {};
    out_neighbors_vec_ = {};

    node_properties_ = torch::Tensor();
}

void GNNGraph::to(torch::Device device, at::cuda::CUDAStream *compute_stream, at::cuda::CUDAStream *transfer_stream) {
    node_ids_ = transfer_tensor(node_ids_, device, compute_stream, transfer_stream);
    hop_offsets_ = transfer_tensor(hop_offsets_, device, compute_stream, transfer_stream);

    out_offsets_ = transfer_tensor(out_offsets_, device, compute_stream, transfer_stream);

    in_offsets_ = transfer_tensor(in_offsets_, device, compute_stream, transfer_stream);

    for (int i = 0; i < in_neighbors_vec_.size(); i++) {
        in_neighbors_vec_[i] = transfer_tensor(in_neighbors_vec_[i], device, compute_stream, transfer_stream);
    }

    for (int i = 0; i < out_neighbors_vec_.size(); i++) {
        out_neighbors_vec_[i] = transfer_tensor(out_neighbors_vec_[i], device, compute_stream, transfer_stream);
    }

    node_properties_ = transfer_tensor(node_properties_, device, compute_stream, transfer_stream);
}

int64_t GNNGraph::getLayerOffset() {
    return hop_offsets_[1].item<int64_t>();
}

void GNNGraph::prepareForNextLayer() {
    int64_t num_nodes_to_remove = (hop_offsets_[1] - hop_offsets_[0]).item<int64_t>();
    int64_t num_finished_nodes = (hop_offsets_[2] - hop_offsets_[1]).item<int64_t>();

    if (src_sorted_edges_.size(0) > 0) {
        int64_t finished_out_neighbors = out_offsets_[num_finished_nodes].item<int64_t>();
        src_sorted_edges_ = src_sorted_edges_.narrow(0, finished_out_neighbors, src_sorted_edges_.size(0) - finished_out_neighbors);
        out_neighbors_mapping_ = out_neighbors_mapping_.narrow(0, finished_out_neighbors, out_neighbors_mapping_.size(0) - finished_out_neighbors) - num_nodes_to_remove;
        out_offsets_ = out_offsets_.narrow(0, num_finished_nodes, out_offsets_.size(0) - num_finished_nodes) - finished_out_neighbors;
    }
    out_num_neighbors_ = out_num_neighbors_.narrow(0, num_finished_nodes, out_num_neighbors_.size(0) - num_finished_nodes);

    if (dst_sorted_edges_.size(0) > 0) {
        int64_t finished_in_neighbors = in_offsets_[num_finished_nodes].item<int64_t>();
        dst_sorted_edges_ = dst_sorted_edges_.narrow(0, finished_in_neighbors, dst_sorted_edges_.size(0) - finished_in_neighbors);
        in_neighbors_mapping_ = in_neighbors_mapping_.narrow(0, finished_in_neighbors, in_neighbors_mapping_.size(0) - finished_in_neighbors) - num_nodes_to_remove;
        in_offsets_ = in_offsets_.narrow(0, num_finished_nodes, in_offsets_.size(0) - num_finished_nodes) - finished_in_neighbors;
    }
    in_num_neighbors_ = in_num_neighbors_.narrow(0, num_finished_nodes, in_num_neighbors_.size(0) - num_finished_nodes);

    node_ids_ = node_ids_.narrow(0, num_nodes_to_remove, node_ids_.size(0) - num_nodes_to_remove);
    hop_offsets_ = hop_offsets_.narrow(0, 1, hop_offsets_.size(0) - 1) - num_nodes_to_remove;
}

Indices GNNGraph::getNeighborIDs(bool incoming, bool global_ids) {
    if (global_ids) {
        // return global node ids
        if (incoming) {
            return dst_sorted_edges_.select(1, 0);
        } else {
            return src_sorted_edges_.select(1, -1);
        }
    } else {
        // return node ids local to the batch
        if (incoming) {
            return in_neighbors_mapping_;
        } else {
            return out_neighbors_mapping_;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GNNGraph::getCombinedNeighborIDs() {
    torch::Tensor new_offsets = in_offsets_ + out_offsets_;
    torch::Tensor new_num_neighbors = in_num_neighbors_ + out_num_neighbors_;
    torch::Tensor new_mapping = torch::empty(in_neighbors_mapping_.size(0) + out_neighbors_mapping_.size(0), in_neighbors_mapping_.options());

    torch::Tensor repeated_starts = new_offsets.repeat_interleave(in_num_neighbors_);
    torch::Tensor repeated_offsets = in_offsets_.repeat_interleave(in_num_neighbors_);
    torch::Tensor arange = torch::arange(repeated_offsets.size(0), repeated_offsets.options());
    torch::Tensor incoming_indices = repeated_starts + arange - repeated_offsets;

    torch::Tensor global_offsets = new_offsets + in_num_neighbors_;
    repeated_starts = global_offsets.repeat_interleave(out_num_neighbors_);
    repeated_offsets = out_offsets_.repeat_interleave(out_num_neighbors_);
    arange = torch::arange(repeated_offsets.size(0), repeated_offsets.options());
    torch::Tensor outgoing_indices = repeated_starts + arange - repeated_offsets;

    new_mapping.index_copy_(0, incoming_indices, in_neighbors_mapping_);
    new_mapping.index_copy_(0, outgoing_indices, out_neighbors_mapping_);

    return std::forward_as_tuple(new_offsets, new_num_neighbors, new_mapping);
}


void GNNGraph::performMap() {
    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids_.device());

    torch::Tensor local_id_to_batch_map = torch::zeros({num_nodes_in_memory_}, device_options);

    local_id_to_batch_map.index_copy_(0, node_ids_, torch::arange(node_ids_.size(0), device_options));

    if (out_neighbors_vec_.size() > 0) {
        src_sorted_edges_ = torch::cat({out_neighbors_vec_}, 0);
        out_neighbors_mapping_ = local_id_to_batch_map.gather(0, src_sorted_edges_.select(1, -1));

        out_neighbors_vec_ = {};

        torch::Tensor tmp_out_offsets = torch::cat({out_offsets_, torch::tensor({src_sorted_edges_.size(0)}, out_offsets_.device())});
        out_num_neighbors_ = tmp_out_offsets.narrow(0, 1, out_offsets_.size(0)) - tmp_out_offsets.narrow(0, 0, out_offsets_.size(0));
    } else {
        out_num_neighbors_ = torch::zeros({node_ids_.size(0)}, device_options);
    }

    if (in_neighbors_vec_.size() > 0) {
        dst_sorted_edges_ = torch::cat({in_neighbors_vec_}, 0);
        in_neighbors_mapping_ = local_id_to_batch_map.gather(0, dst_sorted_edges_.select(1, 0));

        in_neighbors_vec_ = {};

        torch::Tensor tmp_in_offsets = torch::cat({in_offsets_, torch::tensor({dst_sorted_edges_.size(0)}, in_offsets_.device())});
        in_num_neighbors_ = tmp_in_offsets.narrow(0, 1, in_offsets_.size(0)) - tmp_in_offsets.narrow(0, 0, in_offsets_.size(0));
    } else {
        in_num_neighbors_ = torch::zeros({node_ids_.size(0)}, device_options);
    }

    // only works for torch > 1.8
//    in_num_neighbors_ = torch::diff(in_offsets_, 1, 0, {}, torch::tensor({dst_sorted_edges_.size(0)}, in_offsets_.device()));
//    out_num_neighbors_ = torch::diff(out_offsets_, 1, 0, {}, torch::tensor({src_sorted_edges_.size(0)}, out_offsets_.device()));
}

void GNNGraph::setNodeProperties(torch::Tensor node_properties) {
    assert(node_properties.size(0) == node_ids_.size(0));
    node_properties_ = node_properties;
}



