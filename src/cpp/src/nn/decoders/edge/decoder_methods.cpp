//
// Created by Jason Mohoney on 3/31/22.
//

#include "nn/decoders/edge/decoder_methods.h"

std::tuple<torch::Tensor, torch::Tensor> only_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;

    bool has_relations;
    if (edges.size(1) == 3) {
        has_relations = true;
    } else if (edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, edges.select(1, -1));

    torch::Tensor rel_ids;

    if (has_relations) {
        rel_ids = edges.select(1, 1);

        torch::Tensor rels = decoder->select_relations(rel_ids);

        pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);

            inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
    }

    return std::forward_as_tuple(pos_scores, inv_pos_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neg_and_pos_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor negative_edges, torch::Tensor node_embeddings) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    std::tie(pos_scores, inv_pos_scores) = only_pos_forward(decoder, positive_edges, node_embeddings);
    std::tie(neg_scores, inv_neg_scores) = only_pos_forward(decoder, negative_edges, node_embeddings);

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> node_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                            torch::Tensor node_embeddings, torch::Tensor dst_negs,
                                                                                            torch::Tensor src_negs) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    bool has_relations;
    if (positive_edges.size(1) == 3) {
        has_relations = true;
    } else if (positive_edges.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 or 2 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, positive_edges.select(1, -1));

    torch::Tensor rel_ids;

    torch::Tensor dst_neg_embs = node_embeddings.index_select(0, dst_negs.flatten(0, 1)).reshape({dst_negs.size(0), dst_negs.size(1), -1});

    if (has_relations) {
        rel_ids = positive_edges.select(1, 1);

        torch::Tensor rels = decoder->select_relations(rel_ids);
        torch::Tensor adjusted_src = decoder->apply_relation(src, rels);

        pos_scores = decoder->compute_scores(adjusted_src, dst);
        neg_scores = decoder->compute_scores(adjusted_src, dst_neg_embs);

        if (decoder->use_inverse_relations_) {
            torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
            torch::Tensor adjusted_dst = decoder->apply_relation(dst, inv_rels);
            torch::Tensor src_neg_embs = node_embeddings.index_select(0, src_negs.flatten(0, 1)).reshape({src_negs.size(0), src_negs.size(1), -1});

            inv_pos_scores = decoder->compute_scores(adjusted_dst, src);
            inv_neg_scores = decoder->compute_scores(adjusted_dst, src_neg_embs);
        }
    } else {
        pos_scores = decoder->compute_scores(src, dst);
        neg_scores = decoder->compute_scores(src, dst_neg_embs);
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);

        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_corrupt_forward(shared_ptr<EdgeDecoder> decoder, torch::Tensor positive_edges,
                                                                                           torch::Tensor node_embeddings, torch::Tensor neg_rel_ids) {
    torch::Tensor pos_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_neg_scores;

    if (positive_edges.size(1) != 3) {
        throw TensorSizeMismatchException(positive_edges, "Edge list must be a 3 column tensor");
    }

    torch::Tensor src = node_embeddings.index_select(0, positive_edges.select(1, 0));
    torch::Tensor dst = node_embeddings.index_select(0, positive_edges.select(1, -1));

    torch::Tensor rel_ids = positive_edges.select(1, 1);

    torch::Tensor rels = decoder->select_relations(rel_ids);
    torch::Tensor neg_rels = decoder->select_relations(neg_rel_ids);

    pos_scores = decoder->compute_scores(decoder->apply_relation(src, rels), dst);
    neg_scores = decoder->compute_scores(decoder->apply_relation(src, neg_rels), dst);

    if (decoder->use_inverse_relations_) {
        torch::Tensor inv_rels = decoder->select_relations(rel_ids, true);
        torch::Tensor inv_neg_rels = decoder->select_relations(neg_rel_ids, true);

        inv_pos_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_rels), src);
        inv_neg_scores = decoder->compute_scores(decoder->apply_relation(dst, inv_neg_rels), src);
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}