#include "load_files.h"

#include "kernel_naive.h"

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    load_files();
    auto index = preprocess_ann_naive(
        true, /* is_l2 */
        data_base.dim,
        data_base.length,
        data_base.vec
    );
    std::vector<int> result;
    result.resize(100);
    compute_ann_naive(
        data_query.dim,
        data_base.length,
        100,
        data_query.vec,
        data_base.vec,
        result.data(),
        index
    );
    free(index);
    for (int x : result) {
        std::cout << x << std::endl;
    }
    close_files();
    return 0;
}
