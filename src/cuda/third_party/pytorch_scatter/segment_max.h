#pragma once

#include <datatypes.h>

std::tuple<torch::Tensor, torch::Tensor>
segment_max_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out);