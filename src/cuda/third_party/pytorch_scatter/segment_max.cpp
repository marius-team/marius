#include <torch/script.h>
#include "segment_csr_cuda.h"

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
    std::vector<int64_t> result;
    result.reserve(list.size());
    for (size_t i = 0; i < list.size(); i++)
        result.push_back(list[i]);
    return result;
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SegmentMaxCSR : public torch::autograd::Function<SegmentMaxCSR> {
public:
    static variable_list forward(AutogradContext *ctx, Variable src,
                                 Variable indptr,
                                 torch::optional<Variable> optional_out) {
        ctx->saved_data["src_shape"] = src.sizes();
        auto result = segment_csr_cuda(src, indptr, optional_out, "max");
        auto out = std::get<0>(result);
        auto arg_out = std::get<1>(result).value();
        ctx->save_for_backward({indptr, arg_out});
        ctx->mark_non_differentiable({arg_out});
        if (optional_out.has_value())
            ctx->mark_dirty({optional_out.value()});
        return {out, arg_out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto indptr = saved[0];
        auto arg_out = saved[1];
        auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
        src_shape[indptr.dim() - 1] += 1;
        auto grad_in = torch::zeros(src_shape, grad_out.options());
        grad_in.scatter_(indptr.dim() - 1, arg_out, grad_out);
        grad_in =
                grad_in.narrow(indptr.dim() - 1, 0, src_shape[indptr.dim() - 1] - 1);
        return {grad_in, Variable(), Variable()};
    }
};

std::tuple<torch::Tensor, torch::Tensor>
segment_max_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out) {
    auto result = SegmentMaxCSR::apply(src, indptr, optional_out);
    return std::make_tuple(result[0], result[1]);
}
