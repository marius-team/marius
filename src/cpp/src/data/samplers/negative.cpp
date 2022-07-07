//
// Created by Jason Mohoney on 2/8/22.
//

#include "data/samplers/negative.h"

std::tuple<torch::Tensor, torch::Tensor> batch_sample(torch::Tensor edges, int num_negatives, bool inverse) {
    auto device = edges.device();
    int64_t batch_size = edges.size(0);
    Indices sample_edge_id = torch::randint(0, batch_size, {num_negatives}, device).to(torch::kInt64);
    torch::Tensor edge_sample;

    if (inverse) {
        edge_sample = edges.index_select(0, sample_edge_id).select(1, 0);
    } else {
        edge_sample = edges.index_select(0, sample_edge_id).select(1, -1);
    }
    return std::forward_as_tuple(edge_sample, sample_edge_id);
}

torch::Tensor deg_negative_local_filter(torch::Tensor deg_sample_indices, torch::Tensor edges) {
    if (!deg_sample_indices.defined()) {
        torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
        return torch::empty({0, 2}, ind_opts);
    }

    int64_t num_chunks = deg_sample_indices.size(0);
    int64_t chunk_size = ceil((double)edges.size(0) / num_chunks);
    int64_t num_deg_negs = deg_sample_indices.size(1);

    torch::Tensor chunk_ids = deg_sample_indices.div(chunk_size, "trunc");
    torch::Tensor inv_mask = chunk_ids - torch::arange(0, num_chunks, deg_sample_indices.device()).view({num_chunks, -1});
    torch::Tensor mask = (inv_mask == 0);
    torch::Tensor temp_idx = torch::nonzero(mask);
    torch::Tensor id_offsets = deg_sample_indices.flatten(0, 1).index_select(0, temp_idx.select(1, 0) * num_deg_negs + temp_idx.select(1, 1));

    torch::Tensor filter = torch::stack({id_offsets, temp_idx.select(1, 1)}).transpose(0, 1);
    return filter;
}

torch::Tensor compute_filter_corruption(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                        LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (edges.is_cuda()) {
        return compute_filter_corruption_gpu(graph, edges, corruption_nodes, inverse, global, local_filter_mode, deg_sample_indices);
    } else {
        return compute_filter_corruption_cpu(graph, edges, corruption_nodes, inverse, global, local_filter_mode, deg_sample_indices);
    }
}

torch::Tensor compute_filter_corruption_cpu(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                            LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (local_filter_mode == LocalFilterMode::DEG && !global) {
        return deg_negative_local_filter(deg_sample_indices, edges);
    }

    bool has_relations;

    if (edges.dim() == 3) {
        edges = edges.flatten(0, 1);
    } else if (edges.dim() != 2) {
        throw TensorSizeMismatchException(edges, "Edge list must have three (if chunked) or two dimensions");
    }

    if (edges.size(-1) == 3) {
        has_relations = true;
    } else if (edges.size(-1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list tensor must have 3 or 2 columns.");
    }

    int64_t num_chunks = corruption_nodes.size(0);
    int64_t num_edges = edges.size(0);
    int64_t chunk_size = ceil((double)num_edges / num_chunks);

    torch::Tensor all_sorted_edges;
    torch::Tensor all_sorted_nodes;
    torch::Tensor nodes;
    int tup_id;
    int corrupt_id;

    if (inverse) {
        if (has_relations) {
            tup_id = 2;
        } else {
            tup_id = 1;
        }

        corrupt_id = 0;

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            if (graph->all_dst_sorted_edges_.defined()) {
                all_sorted_edges = graph->all_dst_sorted_edges_;
            } else {
                all_sorted_edges = graph->dst_sorted_edges_;
            }

        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();

    } else {
        tup_id = 0;

        if (has_relations) {
            corrupt_id = 2;
        } else {
            corrupt_id = 1;
        }

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            if (graph->all_src_sorted_edges_.defined()) {
                all_sorted_edges = graph->all_src_sorted_edges_;
            } else {
                all_sorted_edges = graph->src_sorted_edges_;
            }
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    }

    std::vector<std::vector<int64_t>> filters(num_edges);

    torch::Tensor starts = torch::searchsorted(all_sorted_nodes, nodes);
    torch::Tensor ends = torch::searchsorted(all_sorted_nodes, nodes + 1);

    auto edges_accessor = edges.accessor<int64_t, 2>();
    auto starts_accessor = starts.accessor<int64_t, 1>();
    auto ends_accessor = ends.accessor<int64_t, 1>();
    auto sorted_edges_accessor = all_sorted_edges.accessor<int64_t, 2>();
    auto negs_accessor = corruption_nodes.accessor<int64_t, 2>();

    if (global) {
#pragma omp parallel for
        for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
            int64_t curr_start = starts_accessor[edge_id];
            int64_t curr_end = ends_accessor[edge_id];

            for (int64_t curr = curr_start; curr < curr_end; curr++) {
                if ((has_relations && sorted_edges_accessor[curr][1] == edges_accessor[edge_id][1]) || !has_relations) {
                    filters[edge_id].emplace_back(sorted_edges_accessor[curr][corrupt_id]);
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
            int64_t curr_start = starts_accessor[edge_id];
            int64_t curr_end = ends_accessor[edge_id];

            int chunk_id = edge_id / chunk_size;

            for (int64_t neg_id = 0; neg_id < corruption_nodes.size(1); neg_id++) {
                int64_t neg_node = negs_accessor[chunk_id][neg_id];

                for (int64_t curr = curr_start; curr < curr_end; curr++) {
                    if (sorted_edges_accessor[curr][corrupt_id] == neg_node) {
                        if ((has_relations && sorted_edges_accessor[curr][1] == edges_accessor[edge_id][1]) || !has_relations) {
                            filters[edge_id].emplace_back(neg_id);
                            break;
                        }
                    }
                }
            }
        }
    }

    int64_t num_filt = 0;

    for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
        num_filt += filters[edge_id].size();
    }

    torch::Tensor filter = torch::empty({num_filt, 2}, torch::kInt64);

    auto filter_accessor = filter.accessor<int64_t, 2>();

    int64_t offset = 0;
    for (int64_t edge_id = 0; edge_id < nodes.size(0); edge_id++) {
        for (int64_t j = 0; j < filters[edge_id].size(); j++) {
            filter_accessor[offset][0] = edge_id;
            filter_accessor[offset][1] = filters[edge_id][j];
            offset++;
        }
    }
    return filter;
}

torch::Tensor compute_filter_corruption_gpu(shared_ptr<MariusGraph> graph, torch::Tensor edges, torch::Tensor corruption_nodes, bool inverse, bool global,
                                            LocalFilterMode local_filter_mode, torch::Tensor deg_sample_indices) {
    if (local_filter_mode == LocalFilterMode::DEG && !global) {
        return deg_negative_local_filter(deg_sample_indices, edges);
    }

    bool has_relations;

    if (edges.dim() == 3) {
        edges = edges.flatten(0, 1);
    } else if (edges.dim() != 2) {
        throw TensorSizeMismatchException(edges, "Edge list must have three (if chunked) or two dimensions");
    }

    if (edges.size(-1) == 3) {
        has_relations = true;
    } else if (edges.size(-1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list tensor must have 3 or 2 columns.");
    }

    int64_t num_chunks = corruption_nodes.size(0);
    int64_t num_edges = edges.size(0);
    int64_t chunk_size = ceil((double)num_edges / num_chunks);

    int64_t negs_per_pos = corruption_nodes.size(1);

    torch::Tensor filter;
    torch::Tensor all_sorted_edges;
    torch::Tensor all_sorted_nodes;
    torch::Tensor nodes;
    int tup_id;
    int corrupt_id;

    if (inverse) {
        if (has_relations) {
            tup_id = 2;
        } else {
            tup_id = 1;
        }

        corrupt_id = 0;

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            all_sorted_edges = graph->all_dst_sorted_edges_;
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    } else {
        tup_id = 0;

        if (has_relations) {
            corrupt_id = 2;
        } else {
            corrupt_id = 1;
        }

        nodes = edges.select(1, tup_id).contiguous();

        if (global) {
            all_sorted_edges = graph->all_src_sorted_edges_;
        } else {
            all_sorted_edges = edges.index_select(0, nodes.argsort());
        }

        all_sorted_nodes = all_sorted_edges.select(1, tup_id).contiguous();
    }

    torch::Tensor starts = torch::searchsorted(all_sorted_nodes, nodes);
    torch::Tensor ends = torch::searchsorted(all_sorted_nodes, nodes + 1);
    torch::Tensor num_neighbors = ends - starts;

    torch::Tensor summed_num_neighbors = num_neighbors.cumsum(0);
    Indices local_offsets = summed_num_neighbors - num_neighbors;

    if (global) {
        torch::Tensor repeated_starts = starts.repeat_interleave(num_neighbors);
        torch::Tensor repeated_offsets = local_offsets.repeat_interleave(num_neighbors);
        torch::Tensor arange = torch::arange(repeated_offsets.size(0), edges.options());
        torch::Tensor sorted_list_idx = repeated_starts + arange - repeated_offsets;

        torch::Tensor batch_neighbors = all_sorted_edges.index_select(0, sorted_list_idx);
        torch::Tensor edge_ids = torch::arange(edges.size(0), edges.options()).repeat_interleave(num_neighbors);

        if (has_relations) {
            torch::Tensor filter_tmp_ids =
                torch::cat({edge_ids.view({-1, 1}), batch_neighbors.select(1, 1).view({-1, 1}), batch_neighbors.select(1, corrupt_id).view({-1, 1})}, 1);
            torch::Tensor rel_ids = edges.select(1, 1).repeat_interleave(num_neighbors);
            torch::Tensor mask = filter_tmp_ids.select(1, 1) == rel_ids;
            filter_tmp_ids = filter_tmp_ids.index_select(0, torch::arange(filter_tmp_ids.size(0), filter_tmp_ids.options()).masked_select(mask));
            filter = torch::cat({filter_tmp_ids.select(1, 0).view({-1, 1}), filter_tmp_ids.select(1, 2).view({-1, 1})}, 1);
        } else {
            filter = torch::cat({edge_ids.view({-1, 1}), batch_neighbors.select(1, corrupt_id).view({-1, 1})}, 1);
        }
    } else {
        // TODO implement local filtering on the GPU, filter needs to be an int64, shape [*, 2], unit tests for this would be good
        // like above when edges are int32 the filter may end up as int32
        //        torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());
        //        filter = torch::empty({0, 2}, ind_opts);
        throw MariusRuntimeException("Local filtering against all edges in the batch not yet supported on GPU.");
    }
    return filter;
}

torch::Tensor apply_score_filter(torch::Tensor scores, torch::Tensor filter) {
    if (filter.defined()) {
        scores.index_put_({filter.select(1, 0), filter.select(1, 1)}, -1e9);
    }
    return scores;
}

CorruptNodeNegativeSampler::CorruptNodeNegativeSampler(int num_chunks, int num_negatives, float degree_fraction, bool filtered,
                                                       LocalFilterMode local_filter_mode) {
    num_chunks_ = num_chunks;
    num_negatives_ = num_negatives;
    degree_fraction_ = degree_fraction;
    filtered_ = filtered;
    local_filter_mode_ = local_filter_mode;

    if (filtered_) {
        num_chunks_ = 1;
        num_negatives_ = -1;
        degree_fraction_ = 0.0;
    }
}

std::tuple<torch::Tensor, torch::Tensor> CorruptNodeNegativeSampler::getNegatives(shared_ptr<MariusGraph> graph, torch::Tensor edges, bool inverse) {
    vector<Indices> ret_indices(num_chunks_);
    vector<Indices> deg_sample_indices_vec(num_chunks_);

    int64_t num_nodes = graph->num_nodes_in_memory_;

    int num_batch = (int)(num_negatives_ * degree_fraction_);
    int num_uni = num_negatives_ - num_batch;

    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(edges.device());

    // sample uniform nodes
    for (int j = 0; j < num_chunks_; j++) {
        if (num_negatives_ != -1) {
            ret_indices[j] = torch::randint(num_nodes, {num_uni}, ind_opts);

            if (degree_fraction_ > 0) {
                auto tup = batch_sample(edges, num_batch, inverse);
                torch::Tensor deg_sample = std::get<0>(tup);
                ret_indices[j] = torch::cat({deg_sample, ret_indices[j]});

                if (local_filter_mode_ == LocalFilterMode::DEG) {
                    torch::Tensor sample_edge_id = std::get<1>(tup);
                    deg_sample_indices_vec[j] = sample_edge_id;
                }
            }
        } else {
            ret_indices[j] = torch::arange(num_nodes, ind_opts);
        }
    }

    torch::Tensor output_ids = torch::stack(ret_indices);
    torch::Tensor deg_sample_indices;
    if (degree_fraction_ > 0 && local_filter_mode_ == LocalFilterMode::DEG) {
        deg_sample_indices = torch::stack(deg_sample_indices_vec);
    }
    torch::Tensor score_filter = compute_filter_corruption(graph, edges, output_ids, inverse, filtered_, local_filter_mode_, deg_sample_indices);
    return std::forward_as_tuple(output_ids, score_filter);
}