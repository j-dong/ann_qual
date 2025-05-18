#pragma once

#include <memory>
#include <string>
#include "load_files.h"
#include "timer.h"

namespace argparse {
struct ArgumentParser;
}

class Index {
public:
    virtual ~Index()=0;
};
inline Index::~Index() {}

typedef std::unique_ptr<Index> preprocess_decl(bool is_l2, RawVectorData *vectors, RawVectorData *learn, argparse::ArgumentParser &parser);

typedef int compute_decl(RawVectorData *vectors, int k, float *query, int *result, Index *index);

typedef void compute_many_decl(RawVectorData *vectors, int k, RawVectorData *queries, int *result, Index *index);

template<compute_decl Func> void compute_many(
        RawVectorData *vectors, int k, RawVectorData *queries,
        int *result, Index *index) {
    size_t total_len = queries->length * (size_t) k;
    __builtin_memset(result, -1, total_len * sizeof *result);
    ScopedTimer timer("compute_many");
    for (size_t i = 0; i < (size_t) queries->length; i++) {
        Func(vectors, k, &queries->at((int) i, 0),
             &result[(size_t) k * i], index);
    }
    double total_ms = timer.get_ms();
    timer.print_timer_message("avg query latency", -1, total_ms / queries->length);
}

typedef void output_stats_decl(RawVectorData *vectors, int k, RawVectorData *queries, int *result, Index *index);

typedef void make_arg_parser_decl(argparse::ArgumentParser &parser);

typedef std::string out_fn_decl(Index *index, argparse::ArgumentParser *parser);

typedef void save_decl(Index *index, argparse::ArgumentParser *parser);

struct KernelFuncs {
    const char *name;
    preprocess_decl *preprocess;
    compute_decl *compute;
    compute_many_decl *compute_many;
    make_arg_parser_decl *make_arg_parser;
    out_fn_decl *out_fn;
    output_stats_decl *output_stats;
    save_decl *save;
};

typedef KernelFuncs get_funcs_decl();
