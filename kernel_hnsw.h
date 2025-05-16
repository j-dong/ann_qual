#pragma once

#include "kernel.h"

preprocess_decl preprocess_ann_hnsw;
compute_decl compute_ann_hnsw;

make_arg_parser_decl make_arg_parser_hnsw;

inline KernelFuncs get_funcs_hnsw() {
    return KernelFuncs {
        .name = "hnsw",
        .preprocess = preprocess_ann_hnsw,
        .compute = compute_ann_hnsw,
        .compute_many = compute_many<compute_ann_hnsw>,
        .make_arg_parser = make_arg_parser_hnsw,
    };
}
