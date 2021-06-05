//
// Created by Jason Mohoney on 2019-11-20.
//

#include "decoder.h"

#include "config.h"

using std::tuple;
using std::make_tuple;
using std::move;
using std::tie;

tuple<torch::Tensor, torch::Tensor> CosineCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

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

    return make_tuple(move(pos_scores), move(neg_scores));
}

tuple<torch::Tensor, torch::Tensor> DotCompare::operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) {

    int num_chunks = negs.size(0);
    int num_pos = src.size(0);
    int num_per_chunk = (int) ceil((float) num_pos / num_chunks);

    // apply relation operator
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

    return make_tuple(move(pos_scores), move(neg_scores));
}

Embeddings HadamardOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs * rels;
}

Embeddings ComplexHadamardOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    int dim = embs.size(1);

    int real_len = dim / 2;
    int imag_len = dim - dim / 2;

    Embeddings real_emb = embs.narrow(1, 0, real_len);
    Embeddings imag_emb = embs.narrow(1, real_len, imag_len);

    Relations real_rel = rels.narrow(1, 0, real_len);
    Relations imag_rel = rels.narrow(1, real_len, imag_len);

    Embeddings out = torch::zeros_like(embs);

    out.narrow(1, 0, real_len) = (real_emb * real_rel) - (imag_emb * imag_rel);
    out.narrow(1, real_len, imag_len) = (real_emb * imag_rel) + (imag_emb * real_rel);

    return out;
}

Embeddings TranslationOperator::operator()(const Embeddings &embs, const Relations &rels) {
    if (!rels.defined()) {
        return embs;
    }
    return embs + rels;
}

Embeddings NoOp::operator()(const Embeddings &embs, const Relations &rels) {
    (void) rels;
    return embs;
}

NodeClassificationDecoder::NodeClassificationDecoder() {}

LinkPredictionDecoder::LinkPredictionDecoder() {}

LinkPredictionDecoder::LinkPredictionDecoder(Comparator *comparator, RelationOperator *relation_operator, LossFunction *loss_function) {
    comparator_ = comparator;
    relation_operator_ = relation_operator;
    loss_function_ = loss_function;
}

void LinkPredictionDecoder::forward(Batch *batch, bool train) {
    torch::Tensor lhs_pos_scores;
    torch::Tensor lhs_neg_scores;
    torch::Tensor rhs_pos_scores;
    torch::Tensor rhs_neg_scores;

    torch::Tensor loss;
    torch::Tensor lhs_loss;
    torch::Tensor rhs_loss;

    // localSample
    batch->localSample();

    // corrupt destination
    Embeddings adjusted_src_pos = (*relation_operator_)(batch->src_pos_embeddings_, batch->src_relation_emebeddings_);
    tie(rhs_pos_scores, rhs_neg_scores) = (*comparator_)(adjusted_src_pos, batch->dst_pos_embeddings_, batch->dst_all_neg_embeddings_);

    // corrupt source
    Embeddings adjusted_dst_pos = (*relation_operator_)(batch->dst_pos_embeddings_, batch->dst_relation_emebeddings_);
    tie(lhs_pos_scores, lhs_neg_scores) = (*comparator_)(adjusted_dst_pos, batch->src_pos_embeddings_, batch->src_all_neg_embeddings_);

    // filter scores
    if (batch->dst_neg_filter_.defined()) {
        rhs_neg_scores.flatten(0, 1).index_fill_(0, batch->dst_neg_filter_, -1e9);
        lhs_neg_scores.flatten(0, 1).index_fill_(0, batch->src_neg_filter_, -1e9);
    }

    if (train) {
        lhs_loss = (*loss_function_)(lhs_pos_scores, lhs_neg_scores);
        rhs_loss = (*loss_function_)(rhs_pos_scores, rhs_neg_scores);
        loss = lhs_loss + rhs_loss;

        if (marius_options.training.regularization_coef > 0) {
            torch::Tensor reg_loss;

            if (marius_options.general.num_relations <= 1) {
                reg_loss = marius_options.training.regularization_coef / 2 * torch::sum(
                    (torch::norm(batch->src_pos_embeddings_, marius_options.training.regularization_norm, 0)
                        + torch::norm(batch->dst_pos_embeddings_, marius_options.training.regularization_norm, 0)));
            } else {
                reg_loss = marius_options.training.regularization_coef / 4 * torch::sum(
                    (torch::norm(batch->src_pos_embeddings_, marius_options.training.regularization_norm, 0)
                        + torch::norm(batch->dst_pos_embeddings_, marius_options.training.regularization_norm, 0)
                        + torch::norm(batch->src_relation_emebeddings_, marius_options.training.regularization_norm, 0)
                        + torch::norm(batch->dst_relation_emebeddings_, marius_options.training.regularization_norm, 0)));
            }

            SPDLOG_DEBUG("Loss: {}, Regularization loss: {}", loss.item<float>(), reg_loss.item<float>());
            loss = loss + reg_loss;
        }

        loss.backward();
    }

    else {
        torch::Tensor lhs_ranks;
        torch::Tensor rhs_ranks;

        torch::Tensor auc;
        torch::Tensor lhs_auc;
        torch::Tensor rhs_auc;

        if (marius_options.evaluation.filtered_evaluation) {
            for (int64_t i = 0; i < batch->batch_size_; i++) {
                lhs_neg_scores[i].index_fill_(0, batch->src_neg_filter_eval_[i].to(batch->src_pos_embeddings_.device()), -1e9);
                rhs_neg_scores[i].index_fill_(0, batch->dst_neg_filter_eval_[i].to(batch->src_pos_embeddings_.device()), -1e9);
            }
        }

        lhs_ranks = (lhs_neg_scores >= lhs_pos_scores.unsqueeze(1)).sum(1) + 1;
        rhs_ranks = (rhs_neg_scores >= rhs_pos_scores.unsqueeze(1)).sum(1) + 1;

        auto auc_opts = torch::TensorOptions().dtype(torch::kInt64).device(batch->src_pos_embeddings_.device());

        lhs_auc = (lhs_pos_scores.index_select(0, torch::randint(lhs_pos_scores.size(0), {marius_options.evaluation.batch_size}, auc_opts))
            > lhs_neg_scores.flatten(0, 1).index_select(0, torch::randint(lhs_neg_scores.flatten(0, 1).size(0), {marius_options.evaluation.batch_size}, auc_opts))).to(marius_options.storage.embeddings_dtype).mean();

        rhs_auc = (rhs_pos_scores.index_select(0, torch::randint(rhs_pos_scores.size(0), {marius_options.evaluation.batch_size}, auc_opts))
            > rhs_neg_scores.flatten(0, 1).index_select(0, torch::randint(rhs_neg_scores.flatten(0, 1).size(0), {marius_options.evaluation.batch_size}, auc_opts))).to(marius_options.storage.embeddings_dtype).mean();

        auc = (lhs_auc + rhs_auc) / 2;

        batch->auc_ = auc;

        batch->ranks_ = torch::cat({lhs_ranks, rhs_ranks});

        batch->host_transfer_.synchronize();
    }
}

DistMult::DistMult() {
    if (marius_options.loss.loss_function_type == LossFunctionType::SoftMax) {
        loss_function_ = new SoftMax(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::RankingLoss) {
        loss_function_ = new RankingLoss(marius_options.loss.margin, marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEAfterSigmoidLoss) {
        loss_function_ = new BCEAfterSigmoidLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEWithLogitsLoss) {
        loss_function_ = new BCEWithLogitsLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::MSELoss) {
        loss_function_ = new MSELoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::SoftPlusLoss) {
        loss_function_ = new SoftPlusLoss(marius_options.loss.reduction_type);
    }
    comparator_ = new DotCompare();
    relation_operator_ = new HadamardOperator();
}

TransE::TransE() {
    if (marius_options.loss.loss_function_type == LossFunctionType::SoftMax) {
        loss_function_ = new SoftMax(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::RankingLoss) {
        loss_function_ = new RankingLoss(marius_options.loss.margin, marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEAfterSigmoidLoss) {
        loss_function_ = new BCEAfterSigmoidLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEWithLogitsLoss) {
        loss_function_ = new BCEWithLogitsLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::MSELoss) {
        loss_function_ = new MSELoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::SoftPlusLoss) {
        loss_function_ = new SoftPlusLoss(marius_options.loss.reduction_type);
    }
    comparator_ = new CosineCompare();
    relation_operator_ = new TranslationOperator();
}

ComplEx::ComplEx() {
    if (marius_options.loss.loss_function_type == LossFunctionType::SoftMax) {
        loss_function_ = new SoftMax(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::RankingLoss) {
        loss_function_ = new RankingLoss(marius_options.loss.margin, marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEAfterSigmoidLoss) {
        loss_function_ = new BCEAfterSigmoidLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::BCEWithLogitsLoss) {
        loss_function_ = new BCEWithLogitsLoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::MSELoss) {
        loss_function_ = new MSELoss(marius_options.loss.reduction_type);
    } else if (marius_options.loss.loss_function_type == LossFunctionType::SoftPlusLoss) {
        loss_function_ = new SoftPlusLoss(marius_options.loss.reduction_type);
    }
    comparator_ = new DotCompare();
    relation_operator_ = new ComplexHadamardOperator();
}

