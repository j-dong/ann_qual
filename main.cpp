#include "load_files.h"

#include "kernel_naive.h"
#include "kernel_pq.h"

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    load_files();
    auto index2 = preprocess_ann_pq(
        true,
        &data_base,
        &data_learn
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
    compute_ann_pq(
        &data_base,
        100,
        &data_query.vec[1],
        result.data(),
        index2.get()
    );
    for (int x : result) {
        std::cout << x << std::endl;
    }
    close_files();
    return 0;
}
