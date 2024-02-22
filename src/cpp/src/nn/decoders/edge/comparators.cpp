//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/decoders/edge/comparators.h"

torch::Tensor pad_and_reshape(torch::Tensor input, int num_chunks) {
    int num_pos = input.size(0);
    int num_per_chunk = (int)ceil((float)num_pos / num_chunks);

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        input = torch::nn::functional::pad(input, options);
    }

    input = input.view({num_chunks, num_per_chunk, input.size(1)});

    return input;
}

torch::Tensor L2Compare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    if (src.sizes() == dst.sizes()) {
        return torch::pairwise_distance(src, dst);
    } else {
        src = pad_and_reshape(src, dst.size(0));

        torch::Tensor x2 = (src.pow(2)).sum(2).unsqueeze(2);
        torch::Tensor y2 = (dst.pow(2)).sum(2).unsqueeze(1);
        torch::Tensor xy = torch::matmul(src, dst.transpose(1, 2));

        double tol = 1e-8;

        // (x - y)^2 = x^2 + y^2 - 2*x*y
        return torch::sqrt(torch::clamp_min(x2 + y2 - 2 * xy, tol)).flatten(0, 1).clone();
    }
}

torch::Tensor CosineCompare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    torch::Tensor src_norm = src.norm(2, -1);
    torch::Tensor dst_norm = dst.norm(2, -1);

    torch::Tensor normalized_src = src * src_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    torch::Tensor normalized_dst = dst * dst_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);

    if (src.sizes() == dst.sizes()) {
        return (src * dst).sum(-1);
    } else {
        src = pad_and_reshape(src, dst.size(0));
        return src.bmm(dst.transpose(-1, -2)).flatten(0, 1);
    }
}

torch::Tensor DotCompare::operator()(torch::Tensor src, torch::Tensor dst) {
    if (!src.defined() || !dst.defined()) {
        throw UndefinedTensorException();
    }

    if (src.sizes() == dst.sizes()) {
        return (src * dst).sum(-1);
    } else {
        src = pad_and_reshape(src, dst.size(0));
        return src.bmm(dst.transpose(-1, -2)).flatten(0, 1);
    }
}