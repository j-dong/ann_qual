#include "kernel.h"

preprocess_decl preprocess_ann_naive;
compute_decl compute_ann_naive;

make_arg_parser_decl make_arg_parser_naive;
out_fn_decl out_fn_naive;

inline KernelFuncs get_funcs_naive() {
    return KernelFuncs {
        .name = "naive",
        .preprocess = preprocess_ann_naive,
        .compute = compute_ann_naive,
        .compute_many = compute_many<compute_ann_naive>,
        .make_arg_parser = make_arg_parser_naive,
        .out_fn = out_fn_naive,
        .output_stats = nullptr,
        .save = nullptr,
    };
}
