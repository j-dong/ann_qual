#include "simd_utils.h"

#include <immintrin.h>
#include <cstdint>
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


float simd_l2dist(float *vec1, float *vec2, int dim) {
    __m256 vsum = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(&vec1[i]),
               b = _mm256_loadu_ps(&vec2[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 dist = _mm256_mul_ps(diff, diff);
        vsum = _mm256_add_ps(vsum, dist);
    }
    __m128 hiQ = _mm256_extractf128_ps(vsum, 1);
    __m128 loQ = _mm256_castps256_ps128(vsum);
    __m128 sumQ = _mm_add_ps(loQ, hiQ);
    __m128 loD = sumQ;
    __m128 hiD = _mm_movehl_ps(sumQ, sumQ);
    __m128 sumD = _mm_add_ps(loD, hiD);
    __m128 lo = sumD, hi = _mm_shuffle_ps(sumD, sumD, 0x1);
    __m128 sum_ = _mm_add_ps(lo, hi);
    float sum = _mm_cvtss_f32(sum_);
    for (; i < dim; i++) {
        float a = vec1[i], b = vec2[i];
        float diff = a - b;
        float dist = diff * diff;
        sum += dist;
    }
    return sum;
}

bool simd_contains(int *arr, int N, int x) {
    int i = 0;
    for (; ((uintptr_t) &arr[i]) % 32; i++) {
        if (arr[i] == x) return true;
    }
    __m256i v = _mm256_set1_epi32(x);
    for (; i + 31 < N; i += 32) {
        __m256i a = _mm256_load_si256((__m256i *) &arr[i]);
        a = _mm256_cmpeq_epi32(v, a);
        __m256i b = _mm256_load_si256((__m256i *) &arr[i + 8]);
        b = _mm256_cmpeq_epi32(v, b);
        __m256i c = _mm256_load_si256((__m256i *) &arr[i + 16]);
        c = _mm256_cmpeq_epi32(v, c);
        __m256i d = _mm256_load_si256((__m256i *) &arr[i + 24]);
        d = _mm256_cmpeq_epi32(v, d);
        __m256i ab = _mm256_or_si256(a, b);
        __m256i cd = _mm256_or_si256(c, d);
        __m256i abcd = _mm256_or_si256(ab, cd);
        if (!_mm256_testz_si256(abcd, abcd)) {
            return true;
        }
    }
    for (; i < N; i++) {
        if (arr[i] == x) return true;
    }
    return false;
}
