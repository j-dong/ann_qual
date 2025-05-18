#pragma once

#include "kernel.h"

preprocess_decl preprocess_ann_hnsw;
compute_decl compute_ann_hnsw;
output_stats_decl output_stats_hnsw;

make_arg_parser_decl make_arg_parser_hnsw;
out_fn_decl out_fn_hnsw;
save_decl save_hnsw;

inline KernelFuncs get_funcs_hnsw() {
    return KernelFuncs {
        .name = "hnsw",
        .preprocess = preprocess_ann_hnsw,
        .compute = compute_ann_hnsw,
        .compute_many = compute_many<compute_ann_hnsw>,
        .make_arg_parser = make_arg_parser_hnsw,
        .out_fn = out_fn_hnsw,
        .output_stats = output_stats_hnsw,
        .save = save_hnsw,
    };
}
