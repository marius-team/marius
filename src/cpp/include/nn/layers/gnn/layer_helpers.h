//
// Created by Jason Mohoney on 10/1/21.
//

#ifndef MARIUS_LAYER_HELPERS_H
#define MARIUS_LAYER_HELPERS_H

#include "common/datatypes.h"

torch::Tensor segment_ids_from_offsets(torch::Tensor offsets, int64_t input_size);

torch::Tensor segmented_sum(torch::Tensor tensor, torch::Tensor segment_ids, int64_t num_segments);

torch::Tensor segmented_sum_with_offsets(torch::Tensor tensor, torch::Tensor offsets);

torch::Tensor segmented_max_with_offsets(torch::Tensor tensor, torch::Tensor offsets);

std::tuple<torch::Tensor, torch::Tensor> attention_softmax(torch::Tensor neighbor_attention, torch::Tensor self_attention, torch::Tensor segment_offsets,
                                                           torch::Tensor segment_ids, torch::Tensor num_nbrs);

#endif  // MARIUS_LAYER_HELPERS_H
