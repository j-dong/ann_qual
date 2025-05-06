#include "kernel.h"

Index *preprocess_ann_naive(bool is_l2, int dim, int num_vectors, float *vectors);

void compute_ann_naive(
    int dim,
    int num_vectors,
    int k,
    float *query,
    float *vectors,
    int *result,
    Index *index
);
