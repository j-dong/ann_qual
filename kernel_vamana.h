#pragma once

#include "kernel.h"

preprocess_decl preprocess_ann_vamana;
compute_decl compute_ann_vamana;
output_stats_decl output_stats_vamana;

make_arg_parser_decl make_arg_parser_vamana;
out_fn_decl out_fn_vamana;
save_decl save_vamana;

inline KernelFuncs get_funcs_vamana() {
    return KernelFuncs {
        .name = "vamana",
        .preprocess = preprocess_ann_vamana,
        .compute = compute_ann_vamana,
        .compute_many = compute_many<compute_ann_vamana>,
        .make_arg_parser = make_arg_parser_vamana,
        .out_fn = out_fn_vamana,
        .output_stats = output_stats_vamana,
        .save = save_vamana,
    };
}
