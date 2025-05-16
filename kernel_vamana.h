#pragma once

#include "kernel.h"

preprocess_decl preprocess_ann_vamana;
compute_decl compute_ann_vamana;

make_arg_parser_decl make_arg_parser_vamana;

inline KernelFuncs get_funcs_vamana() {
    return KernelFuncs {
        .name = "vamana",
        .preprocess = preprocess_ann_vamana,
        .compute = compute_ann_vamana,
        .compute_many = compute_many<compute_ann_vamana>,
        .make_arg_parser = make_arg_parser_vamana,
    };
}
