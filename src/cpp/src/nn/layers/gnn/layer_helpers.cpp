//
// Created by Jason Mohoney on 10/1/21.
//

#include "nn/layers/gnn/layer_helpers.h"

#ifdef MARIUS_CUDA
    #include "pytorch_scatter/segment_max.h"
#endif

torch::Tensor segment_ids_from_offsets(torch::Tensor segment_offsets, int64_t input_size) {
    torch::Tensor segment_ids = torch::zeros({input_size + 1}, segment_offsets.options());
    torch::Tensor ones_tensor = torch::ones({segment_offsets.size(0)}, segment_offsets.options());
    segment_ids.index_add_(0, segment_offsets, ones_tensor);
    segment_ids = segment_ids.cumsum(0) - 1;
    return segment_ids.narrow(0, 0, segment_ids.size(0) - 1);
}

torch::Tensor segmented_sum(torch::Tensor tensor, torch::Tensor segment_ids, int64_t num_segments) {
    auto shape = tensor.sizes().vec();
    shape[0] = num_segments;
    torch::Tensor segsum = torch::zeros(shape, tensor.options());
    segsum.index_add_(0, segment_ids, tensor);
    return segsum;
}

torch::Tensor segmented_sum_with_offsets(torch::Tensor tensor, torch::Tensor segment_offsets) {
    torch::Tensor segment_ids = segment_ids_from_offsets(segment_offsets, tensor.size(0));
    return segmented_sum(tensor, segment_ids, segment_offsets.size(0));
}

torch::Tensor segmented_max_with_offsets(torch::Tensor tensor, torch::Tensor segment_offsets) {
    auto shape = tensor.sizes().vec();
    shape[0] = segment_offsets.size(0);
    torch::Tensor out = torch::zeros(shape, tensor.options());

#ifdef MARIUS_CUDA
    return std::get<0>(segment_max_csr(tensor, torch::cat({segment_offsets, torch::tensor({tensor.size(0)}, segment_offsets.options())}), out));
#else
    return torch::Tensor();
#endif
}

std::tuple<torch::Tensor, torch::Tensor> attention_softmax(torch::Tensor neighbor_attention, torch::Tensor self_attention, torch::Tensor segment_offsets,
                                                           torch::Tensor segment_ids, torch::Tensor num_nbrs) {
    torch::Tensor has_nbrs_mask = torch::not_equal(num_nbrs, 0);
    has_nbrs_mask = has_nbrs_mask.reshape({-1, 1, 1});

    torch::Tensor seg_max = segmented_max_with_offsets(neighbor_attention, segment_offsets);
    torch::Tensor attention_max = torch::where(has_nbrs_mask, torch::maximum(seg_max, self_attention), self_attention);

    self_attention = torch::exp(self_attention - attention_max);

    attention_max = attention_max.index_select(0, segment_ids);
    neighbor_attention = torch::exp(neighbor_attention - attention_max);

    torch::Tensor seg_sum = segmented_sum(neighbor_attention, segment_ids, segment_offsets.size(0));
    torch::Tensor attention_sum = seg_sum + self_attention;

    self_attention = self_attention / attention_sum;

    attention_sum = attention_sum.index_select(0, segment_ids);
    neighbor_attention = neighbor_attention / attention_sum;

    return std::forward_as_tuple(neighbor_attention, self_attention);
}