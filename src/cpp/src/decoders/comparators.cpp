//
// Created by Jason Mohoney on 9/29/21.
//

#include <decoders/comparators.h>

std::tuple<torch::Tensor, torch::Tensor> L2Compare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    Embeddings adjusted_src = src;
    Embeddings adjusted_dst = dst;

    // pad embedding tensor if the number of elements is not divisible by the number of chunks
    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        adjusted_src = torch::nn::functional::pad(adjusted_src, options);
        adjusted_dst = torch::nn::functional::pad(adjusted_dst, options);
    }

    adjusted_src = adjusted_src.unsqueeze(1);
    adjusted_dst = adjusted_dst.unsqueeze(1);

//    // (x - y)^2 = x^2 + y^2 - 2*x*y
//    torch::Tensor x2 = torch::matmul(adjusted_src, adjusted_src.transpose(1, 2));
//    torch::Tensor y2 = torch::matmul(adjusted_dst, adjusted_dst.transpose(1, 2));
//    torch::Tensor xy = torch::matmul(adjusted_src, adjusted_dst.transpose(1, 2));
//
//    // need to clamp_min in order to prevent sqrt by negative number and divide by 0 in backward pass.
//    double tol = 1e-8;
//    torch::Tensor pos_scores = torch::sqrt(torch::clamp_min(x2 + y2 - 2*xy, tol)).flatten(0, -1);
//
//    // do the same for the negatives, but batched
//    auto adjusted_negs = negs.unsqueeze(2);
//
//    x2 = x2.reshape({num_chunks, num_per_chunk}).unsqueeze(-1).expand({num_chunks, num_per_chunk, negs.size(1)});
//    y2 = torch::matmul(adjusted_negs, adjusted_negs.transpose(2, 3)).flatten(1, -1);
//    y2 = y2.unsqueeze(1).expand({num_chunks, num_per_chunk, negs.size(1)});
//    xy = torch::bmm(adjusted_src.reshape({num_chunks, num_per_chunk, -1}), adjusted_negs.flatten(1, 2).transpose(1, 2));
//
//    torch::Tensor neg_scores = torch::sqrt(torch::clamp_min(x2 + y2 - 2*xy, tol)).flatten(0, 1);

//     using cdist throws an error in the backwards pass, use alternative method for computing batched L2 distance
    // compute pairwise distance between source nodes and destination nodes
    torch::Tensor pos_scores = torch::cdist(adjusted_src, adjusted_dst).flatten(0, 2);

    adjusted_src = adjusted_src.view({num_chunks, num_per_chunk, src.size(1)});
    // compute batched distance between source nodes and negative nodes
    torch::Tensor neg_scores = torch::cdist(adjusted_src, negs).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}


std::tuple<torch::Tensor, torch::Tensor> CosineCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int64_t) ceil((float) num_pos / num_chunks);

    torch::Tensor src_norm = src.norm(2, -1);
    torch::Tensor dst_norm = dst.norm(2, -1);
    torch::Tensor neg_norm = negs.norm(2, -1);

    Embeddings normalized_src = src * src_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    Embeddings normalized_dst = dst * dst_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);
    Embeddings normalized_neg = negs * neg_norm.clamp_min(1e-10).reciprocal().unsqueeze(-1);

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        normalized_src = torch::nn::functional::pad(normalized_src, options);
        normalized_dst = torch::nn::functional::pad(normalized_dst, options);
    }

    torch::Tensor pos_scores = (normalized_src * normalized_dst).sum(-1);
    normalized_src = normalized_src.view({num_chunks, num_per_chunk, normalized_src.size(1)});
    torch::Tensor neg_scores = normalized_src.bmm(normalized_neg.transpose(-1, -2)).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}

std::tuple<torch::Tensor, torch::Tensor> DotCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    Embeddings adjusted_src = src;
    Embeddings adjusted_dst = dst;

    if (num_per_chunk != num_pos / num_chunks) {
        int64_t new_size = num_per_chunk * num_chunks;
        torch::nn::functional::PadFuncOptions options({0, 0, 0, new_size - num_pos});
        adjusted_src = torch::nn::functional::pad(adjusted_src, options);
        adjusted_dst = torch::nn::functional::pad(adjusted_dst, options);
    }

    torch::Tensor pos_scores = (adjusted_src * adjusted_dst).sum(-1);
    adjusted_src = adjusted_src.view({num_chunks, num_per_chunk, src.size(1)});

    torch::Tensor neg_scores = adjusted_src.bmm(negs.transpose(-1, -2)).flatten(0, 1);

    return std::make_tuple(std::move(pos_scores), std::move(neg_scores));
}