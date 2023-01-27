//
// Created by Jason Mohoney on 8/25/21.
//

#include "data/graph.h"

#include "common/util.h"
#include "data/samplers/neighbor.h"

#ifdef MARIUS_OMP
#include "omp.h"
#endif

MariusGraph::MariusGraph(){};

MariusGraph::MariusGraph(EdgeList src_sorted_edges, EdgeList dst_sorted_edges, int64_t num_nodes_in_memory) {
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
}

MariusGraph::MariusGraph(EdgeList edges) {
    EdgeList src_sorted_edges = edges.index_select(0, edges.select(1, 0).argsort());
    EdgeList dst_sorted_edges = edges.index_select(0, edges.select(1, -1).argsort());
    int64_t num_nodes_in_memory = std::get<0>(torch::_unique(torch::cat({edges.select(1, 0), edges.select(1, -1)}))).size(0);

    MariusGraph(src_sorted_edges, dst_sorted_edges, num_nodes_in_memory);
}

MariusGraph::~MariusGraph() { clear(); }

Indices MariusGraph::getNodeIDs() { return node_ids_; }

Indices MariusGraph::getEdges(bool incoming) {
    if (incoming) {
        return dst_sorted_edges_;
    } else {
        return src_sorted_edges_;
    }
}

Indices MariusGraph::getRelationIDs(bool incoming) {
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
    all_src_sorted_edges_ = torch::Tensor();
    all_dst_sorted_edges_ = torch::Tensor();
    active_in_memory_subgraph_ = torch::Tensor();
    out_sorted_uniques_ = torch::Tensor();
    out_offsets_ = torch::Tensor();
    out_num_neighbors_ = torch::Tensor();
    in_sorted_uniques_ = torch::Tensor();
    in_offsets_ = torch::Tensor();
    in_num_neighbors_ = torch::Tensor();
    all_src_sorted_edges_ = torch::Tensor();
    all_dst_sorted_edges_ = torch::Tensor();
}

void MariusGraph::to(torch::Device device) {
    node_ids_ = node_ids_.to(device);
    src_sorted_edges_ = src_sorted_edges_.to(device);
    dst_sorted_edges_ = dst_sorted_edges_.to(device);
    out_sorted_uniques_ = out_sorted_uniques_.to(device);
    out_offsets_ = out_offsets_.to(device);
    out_num_neighbors_ = out_num_neighbors_.to(device);
    in_sorted_uniques_ = in_sorted_uniques_.to(device);
    in_offsets_ = in_offsets_.to(device);
}

// 1 hop sampler
std::tuple<torch::Tensor, torch::Tensor> MariusGraph::getNeighborsForNodeIds(torch::Tensor node_ids, bool incoming,
                                                                             NeighborSamplingLayer neighbor_sampling_layer, int max_neighbors_size,
                                                                             float rate) {
    int gpu = 0;

    if (node_ids.is_cuda()) {
        gpu = 1;
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());

    Indices in_memory_ids;
    torch::Tensor mask;
    torch::Tensor num_neighbors = torch::zeros_like(node_ids);
    Indices global_offsets = torch::zeros_like(node_ids);

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

    torch::Tensor summed_num_neighbors = num_neighbors.cumsum(0);
    Indices local_offsets = summed_num_neighbors - num_neighbors;
    int64_t total_neighbors = summed_num_neighbors[-1].item<int64_t>();

    std::tuple<torch::Tensor, torch::Tensor> ret;

    torch::Tensor edges;
    int64_t max_id;

    if (incoming) {
        edges = dst_sorted_edges_;
        max_id = max_in_num_neighbors_;
    } else {
        edges = src_sorted_edges_;
        max_id = max_out_num_neighbors_;
    }

    switch (neighbor_sampling_layer) {
        case NeighborSamplingLayer::ALL: {
            if (gpu) {
                ret = sample_all_gpu(edges, global_offsets, local_offsets, num_neighbors);
            } else {
                ret = sample_all_cpu(edges, global_offsets, local_offsets, num_neighbors, total_neighbors);
            }
            break;
        }
        case NeighborSamplingLayer::UNIFORM: {
            if (gpu) {
                ret = sample_uniform_gpu(edges, global_offsets, local_offsets, num_neighbors, max_neighbors_size, max_id);
            } else {
                ret = sample_uniform_cpu(edges, global_offsets, local_offsets, num_neighbors, max_neighbors_size, total_neighbors);
            }
            break;
        }
        case NeighborSamplingLayer::DROPOUT: {
            if (gpu) {
                ret = sample_dropout_gpu(edges, global_offsets, local_offsets, num_neighbors, rate);
            } else {
                ret = sample_dropout_cpu(edges, global_offsets, local_offsets, num_neighbors, rate, total_neighbors);
            }
            break;
        }
    }
    return ret;
}

void MariusGraph::sortAllEdges(EdgeList all_edges) {
    all_src_sorted_edges_ = all_edges.index_select(0, all_edges.select(1, 0).argsort(0, false)).to(torch::kInt64);
    all_dst_sorted_edges_ = all_edges.index_select(0, all_edges.select(1, -1).argsort(0, false)).to(torch::kInt64);
}

DENSEGraph::DENSEGraph(){};

DENSEGraph::DENSEGraph(Indices hop_offsets, Indices node_ids, Indices in_offsets, std::vector<torch::Tensor> in_neighbors_vec, Indices in_neighbors_mapping,
                       Indices out_offsets, std::vector<torch::Tensor> out_neighbors_vec, Indices out_neighbors_mapping, int num_nodes_in_memory) {
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

DENSEGraph::~DENSEGraph() { clear(); }

void DENSEGraph::clear() {
    MariusGraph::clear();

    hop_offsets_ = torch::Tensor();

    in_neighbors_mapping_ = torch::Tensor();
    out_neighbors_mapping_ = torch::Tensor();

    in_neighbors_vec_ = {};
    out_neighbors_vec_ = {};

    node_properties_ = torch::Tensor();
}

void DENSEGraph::to(torch::Device device) {
    node_ids_ = node_ids_.to(device);
    hop_offsets_ = hop_offsets_.to(device);

    if (out_offsets_.defined()) {
        out_offsets_ = out_offsets_.to(device);
    }

    if (in_offsets_.defined()) {
        in_offsets_ = in_offsets_.to(device);
    }

    for (int i = 0; i < in_neighbors_vec_.size(); i++) {
        in_neighbors_vec_[i] = in_neighbors_vec_[i].to(device);
    }

    for (int i = 0; i < out_neighbors_vec_.size(); i++) {
        out_neighbors_vec_[i] = out_neighbors_vec_[i].to(device);
    }

    if (node_properties_.defined()) {
        node_properties_ = node_properties_.to(device);
    }
}

int64_t DENSEGraph::getLayerOffset() { return hop_offsets_[1].item<int64_t>(); }

void DENSEGraph::prepareForNextLayer() {
    int64_t num_nodes_to_remove = (hop_offsets_[1] - hop_offsets_[0]).item<int64_t>();
    int64_t num_finished_nodes = (hop_offsets_[2] - hop_offsets_[1]).item<int64_t>();

    if (src_sorted_edges_.size(0) > 0) {
        if (num_finished_nodes == out_offsets_.size(0)) {
            return;
        }
        int64_t finished_out_neighbors = out_offsets_[num_finished_nodes].item<int64_t>();
        src_sorted_edges_ = src_sorted_edges_.narrow(0, finished_out_neighbors, src_sorted_edges_.size(0) - finished_out_neighbors);
        out_neighbors_mapping_ =
            out_neighbors_mapping_.narrow(0, finished_out_neighbors, out_neighbors_mapping_.size(0) - finished_out_neighbors) - num_nodes_to_remove;
        out_offsets_ = out_offsets_.narrow(0, num_finished_nodes, out_offsets_.size(0) - num_finished_nodes) - finished_out_neighbors;
    }
    out_num_neighbors_ = out_num_neighbors_.narrow(0, num_finished_nodes, out_num_neighbors_.size(0) - num_finished_nodes);

    if (dst_sorted_edges_.size(0) > 0) {
        if (num_finished_nodes == in_offsets_.size(0)) {
            return;
        }
        int64_t finished_in_neighbors = in_offsets_[num_finished_nodes].item<int64_t>();
        dst_sorted_edges_ = dst_sorted_edges_.narrow(0, finished_in_neighbors, dst_sorted_edges_.size(0) - finished_in_neighbors);
        in_neighbors_mapping_ =
            in_neighbors_mapping_.narrow(0, finished_in_neighbors, in_neighbors_mapping_.size(0) - finished_in_neighbors) - num_nodes_to_remove;
        in_offsets_ = in_offsets_.narrow(0, num_finished_nodes, in_offsets_.size(0) - num_finished_nodes) - finished_in_neighbors;
    }
    in_num_neighbors_ = in_num_neighbors_.narrow(0, num_finished_nodes, in_num_neighbors_.size(0) - num_finished_nodes);

    node_ids_ = node_ids_.narrow(0, num_nodes_to_remove, node_ids_.size(0) - num_nodes_to_remove);
    hop_offsets_ = hop_offsets_.narrow(0, 1, hop_offsets_.size(0) - 1) - num_nodes_to_remove;
}

Indices DENSEGraph::getNeighborIDs(bool incoming, bool global_ids) {
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

void DENSEGraph::performMap() {
    if (!node_ids_.defined()) {
        return;
    }

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

void DENSEGraph::setNodeProperties(torch::Tensor node_properties) {
    assert(node_properties.size(0) == node_ids_.size(0));
    node_properties_ = node_properties;
}
