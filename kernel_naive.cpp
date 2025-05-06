#include "kernel_naive.h"
#include "cblas.h"

#include <vector>
#include <algorithm>

class NaiveIndex : public Index {
public:
    float *data;
    ~NaiveIndex() { delete[] data; }
};

Index *preprocess_ann_naive(bool is_l2, int dim, int num_vectors, float *vectors) {
    if (!is_l2) return nullptr;
    NaiveIndex *ret = new NaiveIndex();
    ret->data = new float[num_vectors];
    for (int i = 0; i < num_vectors; i++) {
        ret->data[i] = -0.5f * cblas_sdot(
            dim,
            vectors + 1 + i * (dim + 1),
            1,
            vectors + 1 + i * (dim + 1),
            1
        );
    }
    return ret;
}

void compute_ann_naive(
    int dim,
    int num_vectors,
    int k,
    float *query,
    float *vectors,
    int *result,
    Index *index
) {
    std::vector<float> temp_iprods;
    std::vector<int> temp_indices;
    temp_iprods.resize(num_vectors);
    if (index) {
        memcpy(temp_iprods.data(), ((NaiveIndex *) index)->data, sizeof(float) * num_vectors);
    }
    temp_indices.reserve(num_vectors);
    for (int i = 0; i < num_vectors; i++) temp_indices.push_back(i);
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        num_vectors,
        dim,
        1.0f,
        vectors + 1,
        dim + 1,
        query + 1,
        1,
        index ? 1.0f : 0.0f,
        &temp_iprods[0],
        1
    );
    std::partial_sort(temp_indices.begin(),
                      temp_indices.begin() + k,
                      temp_indices.end(),
                      [&temp_iprods](int a, int b) { return temp_iprods[a] > temp_iprods[b]; });
    memcpy(result, temp_indices.data(), sizeof(int) * k);
}
