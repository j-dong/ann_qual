#include "kernel.h"

preprocess_decl preprocess_ann_pq;
compute_decl compute_ann_pq;

make_arg_parser_decl make_arg_parser_pq;
out_fn_decl out_fn_pq;

inline KernelFuncs get_funcs_pq() {
    return KernelFuncs {
        .name = "pq",
        .preprocess = preprocess_ann_pq,
        .compute = compute_ann_pq,
        .compute_many = compute_many<compute_ann_pq>,
        .make_arg_parser = make_arg_parser_pq,
        .out_fn = out_fn_pq,
        .output_stats = nullptr,
    };
}
