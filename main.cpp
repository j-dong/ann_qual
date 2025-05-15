#include "load_files.h"

#include "kernel_naive.h"
#include "kernel_pq.h"
#include "kernel_hnsw.h"
#include "kernel_vamana.h"
#include "timer.h"

#include <iostream>
#include <vector>
#include <algorithm>

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    load_files();
    // auto index2 = preprocess_ann_pq(
    //     true,
    //     &data_base,
    //     &data_base
    //     // &data_learn
    // );
    // auto index3 = preprocess_ann_hnsw(
    //     true,
    //     &data_base,
    //     &data_base
    //     // &data_learn
    // );
    auto index4 = preprocess_ann_vamana(
        true,
        &data_base,
        &data_base
        // &data_learn
    );
    // auto index = preprocess_ann_naive(
    //     true,
    //     &data_base,
    //     &data_learn
    // );
    std::vector<int> result;
    result.resize(100);
    // compute_ann_naive(
    //     &data_base,
    //     100,
    //     &data_query.vec[1],
    //     result.data(),
    //     index.get()
    // );
    int trials = 100;
    int recall = 0;
    for (int i = 0; i < trials; i++) {
        {
            ScopedTimer timer("ANN", i);
            compute_ann_vamana(
                &data_base,
                100,
                &data_query.vec[1 + i * (data_query.dim + 1)],
                result.data(),
                index4.get()
            );
        }
        int truth = data_ground.at(i, 0);
        if (std::find(result.begin(), result.end(), truth) != result.end()) {
            recall++;
        }
    }
    std::cout << "recall@100: " << (double) recall / trials << std::endl;
    close_files();
    return 0;
}
