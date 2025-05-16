#include "load_files.h"

#include "kernel_naive.h"
#include "kernel_pq.h"
#include "kernel_hnsw.h"
#include "kernel_vamana.h"
#include "simd_utils.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include "control_threading.h"

#include "argparse/argparse.hpp"

struct ManualArrayDeleter {
    size_t size;
    ManualArrayDeleter() : size(0) {}
    ManualArrayDeleter(size_t size) : size(size) {}

    void operator()(argparse::ArgumentParser *p) {
        for (size_t i = size; i > 0; i--) {
            p[i - 1].~ArgumentParser();
        }
        operator delete[](p);
    }
};

int main(int argc, char **argv) {
    std::vector<KernelFuncs> funcs;

    funcs.push_back(get_funcs_naive());
    funcs.push_back(get_funcs_pq());
    funcs.push_back(get_funcs_hnsw());
    funcs.push_back(get_funcs_vamana());

    argparse::ArgumentParser parser;
    parser.add_argument("-p", "--path")
        .help("path prefix for vector data; required files are PREFIX_base.fvecs, PREFIX_learn.fvecs, PREFIX_query.fvecs, PREFIX_groundtruth.ivecs")
        .default_value("G:\\vectors\\siftsmall\\siftsmall");
    parser.add_argument("-k", "--num-results")
        .help("number of results to return")
        .default_value(100)
        .scan<'d', int>();

    std::unique_ptr<argparse::ArgumentParser[], ManualArrayDeleter> parsers;
    {
        // I mean I could just have a vector of pointers
        // but placement new is fun
        argparse::ArgumentParser *temp =
            (argparse::ArgumentParser *)
            operator new[](sizeof(argparse::ArgumentParser) * funcs.size());

        for (size_t ki = 0; ki < funcs.size(); ki++) {
            try {
                new (&temp[ki]) argparse::ArgumentParser(funcs[ki].name);
                try {
                    funcs[ki].make_arg_parser(temp[ki]);
                    parser.add_subparser(temp[ki]);
                } catch (...) {
                    temp[ki].~ArgumentParser();
                    throw;
                }
            } catch (...) {
                for (size_t kj = ki; kj > 0; kj--) {
                    temp[kj - 1].~ArgumentParser();
                }
                operator delete[](temp);
                throw;
            }
        }
        parsers = decltype(parsers)(temp, ManualArrayDeleter(funcs.size()));
    }

    parser.parse_args(argc, argv);

    load_files(parser.get<std::string>("--path"));

    bool handled = false;
    for (size_t ki = 0; ki < funcs.size(); ki++) {
        if (!parser.is_subcommand_used(parsers[ki])) {
            continue;
        }
        handled = true;

        enable_blas_threading();

        auto index = funcs[ki].preprocess(
            true,
            &data_base,
            &data_learn,
            parsers[ki]
        );

        int k = parser.get<int>("-k");

        std::vector<int> result;
        result.resize((size_t) k * data_query.length);

        disable_blas_threading();

        funcs[ki].compute_many(
            &data_base,
            k,
            &data_query,
            result.data(),
            index.get()
        );

        int recall = 0;
        for (size_t i = 0; i < (size_t) data_query.length; i++) {
            int truth = data_ground.at(i, 0);
            if (simd_contains(&result[i * (size_t) k], k, truth)) {
                recall++;
            }
        }

        std::cout << std::endl;
        std::cout << "recall@" << k << ": " << std::setprecision(6)
            << (double) recall / data_query.length << std::endl;
    }

    if (!handled) {
        std::cerr << "error: missing kernel to run" << std::endl;
    }

    close_files();
    return 0;
}
