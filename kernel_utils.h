#pragma once

#include <memory>

struct RawVectorData;

std::unique_ptr<float[]> preprocess_l2_bias(bool is_l2, int dim, int num_vectors, int stride, float *vectors);
std::unique_ptr<float[]> preprocess_l2_bias(bool is_l2, RawVectorData *vectors);

void compute_l2_bias(int dim, int num_vectors, float *vectors, int stride, float *out);
