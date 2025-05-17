#include "kernel_utils.h"
#include "load_files.h"

#include "inc_cblas.h"

std::unique_ptr<float[]> preprocess_l2_bias(bool is_l2, int dim, int num_vectors, int stride, float *vectors) {
    if (!is_l2) return nullptr;
    float *ret = new float[num_vectors];
    compute_l2_bias(dim, num_vectors, vectors, stride, ret);
    return std::unique_ptr<float[]>(ret);
}

std::unique_ptr<float[]> preprocess_l2_bias(bool is_l2, RawVectorData *vectors) {
    return preprocess_l2_bias(is_l2, vectors->dim, vectors->length, vectors->dim + 1, vectors->vec + 1);
}

void compute_l2_bias(int dim, int num_vectors, float *vectors, int stride, float *out) {
    for (int i = 0; i < num_vectors; i++) {
        out[i] = -0.5f * cblas_sdot(
            dim,
            vectors + i * stride,
            1,
            vectors + i * stride,
            1
        );
    }
}
