#include "simd_utils.h"

#include <immintrin.h>
#include <limits>

int simd_argmax(float *vec, int dim) {
    // found https://en.algorithmica.org/hpc/algorithms/argmin/
    float val = -std::numeric_limits<float>::infinity();
    int idx = 0;
    int i = 0;
    // prolog
    for (; i < dim && (uintptr_t) (vec + i) % 32 != 0; i++) {
        if (vec[i] > val) {
            val = vec[i];
            idx = i;
        }
    }
    // vectorized
    __m256 p = _mm256_set1_ps(val);

    for (; i + 31 < dim; i += 32) {
        __m256 y1 = _mm256_load_ps(&vec[i]);
        __m256 y2 = _mm256_load_ps(&vec[i + 8]);
        __m256 y3 = _mm256_load_ps(&vec[i + 16]);
        __m256 y4 = _mm256_load_ps(&vec[i + 24]);
        y1 = _mm256_max_ps(y1, y2);
        y3 = _mm256_max_ps(y3, y4);
        y1 = _mm256_max_ps(y1, y3);
        __m256 mask = _mm256_cmp_ps(p, y1, _CMP_LT_OQ);
        if (!_mm256_testz_ps(mask, mask)) { [[unlikely]]
            idx = i;
            for (int j = i; j < i + 32; j++) {
                val = vec[j] > val ? vec[j] : val;
            }
            p = _mm256_set1_ps(val);
        }
    }

    int end = idx + 32;
    for (int j = idx; j < end && j < dim; j++) {
        if (vec[j] == val) {
            idx = j;
            break;
        }
    }

    // epilog
    for (; i < dim; i++) {
        if (vec[i] > val) {
            val = vec[i];
            idx = i;
        }
    }

    return idx;
}

void simd_pfxsum(int *arr, int N) {
    int sum = 0;
    int i = 0;
    for (; ((uintptr_t) &arr[i]) % 32; i++) {
        sum += arr[i];
        arr[i] = sum;
    }
    for (; i + 7 < N; i += 8) {
        __m256i b = _mm256_set1_epi32(sum);
        __m256i v = _mm256_load_si256((__m256i *) &arr[i]);
        v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
        v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
        __m128i lo = _mm_shuffle_epi32(_mm256_castsi256_si128(v),
                                       _MM_SHUFFLE(3, 3, 3, 3));
        lo = _mm_add_epi32(_mm256_castsi256_si128(b), lo);
        int sum_lo = _mm_cvtsi128_si32(lo);
        int sum_hi = _mm256_extract_epi32(v, 7);
        b = _mm256_inserti128_si256(b, lo, 1);
        v = _mm256_add_epi32(v, b);
        sum = sum_lo + sum_hi;
        // sum = _mm256_extract_epi32(v, 7);
        _mm256_store_si256((__m256i *) &arr[i], v);
    }
    for (; i < N; i++) {
        sum += arr[i];
        arr[i] = sum;
    }
}

