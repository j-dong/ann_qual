#pragma once

int simd_argmax(float *vec, int dim);
void simd_pfxsum(int *arr, int N);
float simd_l2dist(float *vec1, float *vec2, int dim);
