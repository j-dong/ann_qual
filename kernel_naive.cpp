#include "kernel_naive.h"
#include "kernel_utils.h"
#include "load_files.h"

#include "cblas.h"

#include <vector>
#include <algorithm>
#include <cstring>

namespace {
class NaiveIndex : public Index {
public:
    std::unique_ptr<float[]> data;
    ~NaiveIndex() {}
};
}

std::unique_ptr<Index> preprocess_ann_naive(bool is_l2, RawVectorData *vectors, [[maybe_unused]] RawVectorData *learn) {
    auto ret = std::make_unique<NaiveIndex>();
    ret->data = preprocess_l2_bias(is_l2, vectors);
    return ret;
}

void compute_ann_naive(
    RawVectorData *vectors,
    int k,
    float *query,
    int *result,
    Index *index
) {
    std::vector<float> temp_iprods;
    std::vector<int> temp_indices;
    int num_vectors = vectors->length;
    temp_iprods.resize(num_vectors);
    NaiveIndex *nindex = (NaiveIndex *) index;
    if (nindex && nindex->data) {
        memcpy(temp_iprods.data(), nindex->data.get(), sizeof(float) * num_vectors);
    } else { nindex = nullptr; }
    temp_indices.reserve(num_vectors);
    for (int i = 0; i < num_vectors; i++) temp_indices.push_back(i);
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        num_vectors,
        vectors->dim,
        1.0f,
        vectors->vec + 1,
        vectors->dim + 1,
        query,
        1,
        nindex ? 1.0f : 0.0f,
        &temp_iprods[0],
        1
    );
    std::partial_sort(temp_indices.begin(),
                      temp_indices.begin() + k,
                      temp_indices.end(),
                      [&temp_iprods](int a, int b) { return temp_iprods[a] > temp_iprods[b]; });
    memcpy(result, temp_indices.data(), sizeof(int) * k);
}
