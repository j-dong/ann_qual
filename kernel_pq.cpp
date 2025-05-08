#include "kernel_pq.h"
#include "kernel_utils.h"
#include "load_files.h"
#include "timer.h"

#ifndef NOMINMAX
# define NOMINMAX 1
#endif
#include "cblas.h"

#include <cstring>
#include <random>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include <numeric>

template<class T>
using box = std::unique_ptr<T>;

namespace {
struct aligned_deleter {
    void operator()(char *p) {
        ::operator delete[](p, std::align_val_t(64));
    }
};

struct PQIndex : public Index {
    bool is_l2;
    int dim;
    int num_clusters;
    int fine_bits;
    int num_groups;
    int window; // number of clusters to search
    box<float[]> bias;
    box<float[]> clusters;
    box<float[]> codebooks;
    box<float[]> codebooks_bias;
    box<float[]> transform;
    // cluster_start[i] = index of start of ith cluster
    // size is num_clusters + 1, so last element is total size
    box<int[]> cluster_start;
    // indices of vectors for each cluster
    box<int[]> cluster_values;
    // grouped like cluster_values
    // stores quantized indices for each vector
    // value size is ceil(fine_bits / 8) bytes
    // each vector is padded to at least 64 bytes
    // to minimize false sharing
    std::unique_ptr<char[], aligned_deleter> clustered_quant;

    float *get_codebook(int i) {
        int group_dim = (dim + num_groups - 1) / num_groups;
        size_t subcodebook_size = 1 << fine_bits;
        return &codebooks[i * subcodebook_size * group_dim];
    }

    ~PQIndex() {}
};

struct WindowResult {
    float iprod;
    int index;

    bool operator<(const WindowResult &o) const {
        return iprod > o.iprod;
    }
};

constexpr int byte_size(int bits) {
    return (bits + 7) / 8;
}

constexpr int roundup_line(int bytes) {
    return (bytes + 63) / 64 * 64;
}

constexpr int BLOCK_SIZE = 1024;
}

template<bool LOG=true>
static void compute_k_means(int num_clusters, int dim, int stride, int num_vectors, float *vectors, float *bias, float *out_clusters, box<int[]> *out_assignments);
static void compute_assignments(
    int num_vectors,
    int dim,
    int stride,
    float *vectors,
    float *bias,
    int num_clusters,
    float *clusters,
    int *clusters_size,
    float *clusters_bias,
    float *iprods, // scratch
    int *assignments,
    float *energy,
    int *changed
);
static box<float[]> transform_fine_vectors(RawVectorData *vectors, int num_clusters, float *clusters, int *assignments, float *transform);
static void generate_transform(float *mat, int dim);
static void simd_pfxsum(int *arr, int N);
static void write_assignments(int num_vectors, int *assignments, int *gather, int bits, int num_groups, int dim_i, char *out);
static void search_cluster(int c, WindowResult *out, float cluster_iprod, float *iprods, PQIndex *idx);

box<Index> preprocess_ann_pq(bool is_l2, RawVectorData *vectors, RawVectorData *learn) {
    box<PQIndex> ret = std::make_unique<PQIndex>();
    int dim = vectors->dim;
    ret->dim = dim;
    auto learn_bias = preprocess_l2_bias(true, learn);
    auto force_bias = preprocess_l2_bias(true, vectors);
    // ret->num_clusters = (int) std::sqrt(learn->length);
    ret->num_clusters = 8192;
    ret->window = 8;
    ret->fine_bits = 8;
    ret->num_groups = 16;
    // ret->num_clusters = 128;
    // ret->num_clusters = 8192;
    box<int[]> learn_assignments;
    {
        ScopedTimer timer("train coarse quantizer");
        ret->clusters = std::make_unique<float[]>(ret->num_clusters * dim);
        compute_k_means(ret->num_clusters, dim, dim + 1, learn->length, learn->vec + 1, learn_bias.get(), ret->clusters.get(), &learn_assignments);
    }
    {
        ScopedTimer timer("train fine quantizer");
        ret->transform = std::make_unique<float[]>(dim * dim);
        generate_transform(ret->transform.get(), dim);
        box<float[]> transformed_learn = transform_fine_vectors(learn, ret->num_clusters, ret->clusters.get(), learn_assignments.get(), ret->transform.get());
        // ret->fine_bits = 8;
        int subcodebook_size = 1 << ret->fine_bits;
        int group_dim = (dim + ret->num_groups - 1) / ret->num_groups;
        ret->codebooks = std::make_unique<float[]>(
            (size_t) ret->num_groups *
            (size_t) subcodebook_size *
            group_dim
        );
        box<float[]> sub_biases = std::make_unique<float[]>(
            ret->num_groups
            * learn->length
        );
        memset(sub_biases.get(), 0, ret->num_groups * learn->length * sizeof(float));
        for (int i = 0; i < learn->length; i++) {
            float *vec = &transformed_learn[i * dim];
            for (int j = 0; j < dim; j++) {
                int g = j / group_dim;
                sub_biases[i + g * learn->length] += -0.5 * vec[j] * vec[j];
            }
        }
        ret->codebooks_bias = std::make_unique<float[]>(
            ret->num_groups * subcodebook_size
        );
        for (int i = 0; i < ret->num_groups; i++) {
            int start_dim = i * group_dim;
            int cur_dim = std::min(group_dim, dim - start_dim);
            compute_k_means<false>(
                subcodebook_size,
                cur_dim,
                dim,
                learn->length,
                &transformed_learn[start_dim],
                &sub_biases[i * learn->length],
                &ret->codebooks[i * (size_t) subcodebook_size * group_dim],
                nullptr
            );
        }
        memset(ret->codebooks_bias.get(), 0, ret->num_groups * subcodebook_size * sizeof(float));
        for (int i = 0; i < ret->num_groups * subcodebook_size; i++) {
            for (int j = 0; j < group_dim; j++) {
                float val = ret->codebooks[i * group_dim + j];
                ret->codebooks_bias[i] -= 0.5 * val * val;
            }
        }
    }
    {
        ScopedTimer timer("quantize vectors");
        // quantize the vectors
        box<float[]> iprods = std::make_unique<float[]>(ret->num_clusters * BLOCK_SIZE);
        box<int[]> assignments = std::make_unique<int[]>(vectors->length);
        ret->cluster_start = std::make_unique<int[]>(ret->num_clusters + 1);
        ret->cluster_values = std::make_unique<int[]>(vectors->length);
        auto clusters_bias = preprocess_l2_bias(true, dim, ret->num_clusters, dim, ret->clusters.get());
        compute_assignments(
            vectors->length,
            dim,
            dim + 1,
            vectors->vec + 1,
            force_bias.get(),
            ret->num_clusters,
            ret->clusters.get(),
            &ret->cluster_start[1], // offset by 1 so we can use an inclusive scan
            clusters_bias.get(),
            iprods.get(),
            assignments.get(),
            nullptr,
            nullptr
        );
        simd_pfxsum(ret->cluster_start.get(), ret->num_clusters + 1);
        auto scatter = std::make_unique<int[]>(vectors->length);
        auto gather = ret->cluster_values.get();
        {
            auto temp_index = std::make_unique<int[]>(ret->num_clusters);
            memcpy(temp_index.get(), ret->cluster_start.get(), ret->num_clusters * sizeof(float));
            for (int i = 0; i < vectors->length; i++) {
                int j = temp_index[assignments[i]]++;
                scatter[i] = j;
                gather[j] = i;
            }
        }
        auto fine = transform_fine_vectors(
            vectors,
            ret->num_clusters,
            ret->clusters.get(),
            assignments.get(),
            ret->transform.get()
        );
        int subcodebook_size = 1 << ret->fine_bits;
        int group_dim = (dim + ret->num_groups - 1) / ret->num_groups;
        box<float[]> sub_biases = std::make_unique<float[]>(vectors->length);
        box<float[]> sub_iprods = std::make_unique<float[]>(subcodebook_size * BLOCK_SIZE);
        box<int[]> sub_assignments = std::make_unique<int[]>(vectors->length);
        int group_size = byte_size(ret->fine_bits);
        int qvec_size = roundup_line(group_size * ret->num_groups);
        ret->clustered_quant = std::unique_ptr<char[], aligned_deleter>(
            new (std::align_val_t(64)) char[qvec_size * vectors->length]
        );
        box<float[]> qerr = std::make_unique<float[]>(vectors->length);
        box<float[]> qtemp = std::make_unique<float[]>(dim);
        for (int i = 0; i < ret->num_groups; i++) {
            int start_dim = i * group_dim;
            int cur_dim = std::min(group_dim, dim - start_dim);
            for (int i = 0; i < vectors->length; i++) {
                float *vec = vectors->vec + 1 + i * (dim + 1);
                for (int k = 0; k < cur_dim; k++) {
                    int j = start_dim + k;
                    sub_biases[i] += -0.5 * vec[j] * vec[j];
                }
            }
            compute_assignments(
                vectors->length,
                cur_dim,
                dim,
                &fine[start_dim],
                sub_biases.get(),
                subcodebook_size,
                &ret->codebooks[i * (size_t) subcodebook_size * group_dim],
                nullptr,
                clusters_bias.get(),
                sub_iprods.get(),
                sub_assignments.get(),
                nullptr,
                nullptr
            );
            for (int k = 0; k < 20 && i < 10; k++) {
                std::cout << "d" << i << "[" << k << "] -> #" << sub_assignments[k] << std::endl;
            }
            write_assignments(
                vectors->length,
                sub_assignments.get(),
                scatter.get(),
                ret->fine_bits,
                ret->num_groups,
                i,
                ret->clustered_quant.get()
            );
        }
        // print the first codebook
        for (int i = 0; i < subcodebook_size; i++) {
            std::cout << "codebook1[" << i << "] =";
            for (int j = 0; j < group_dim; j++) {
                std::cout << " " << ret->get_codebook(0)[j + i * group_dim];
            }
            std::cout << "\n";
        }
        int cc = 0;
        double total_err = 0.0;
        double coarse_err = 0.0;
        for (int i = 0; i < vectors->length; i++) {
            while (cc < ret->num_clusters && ret->cluster_start[cc + 1] <= i) cc++;
            if (cc == ret->num_clusters) {
                std::cout << "??? we're out of clusters...?" << std::endl;
                std::cout << "we're at " << i << "/" << vectors->length << std::endl;
                for (int j = 0; j <= ret->num_clusters; j++) std::cout << " " << ret->cluster_start[j];
                std::cout << "\n";
                break;
            }
            int idx = gather[i];
            float error = 0.0;
            for (int j = 0; j < dim; j++) {
                float src = vectors->at(idx, j);
                float clust = ret->clusters[cc * dim + j];
                int g = j / group_dim;
                int val = 0;
                int position = qvec_size * i + 1 * g;
                memcpy(&val, &ret->clustered_quant[position], 1);
                float quant = ret->get_codebook(g)[group_dim * val + (j % group_dim)];
                float diff = (clust + quant) - src;
                if (i < 20 && j < 5) {
                    std::cout << "index " << i << " -> " << idx << " in cluster " << cc << "; src = " << src << ", clust = " << clust << std::endl;
                    std::cout << "  g = " << g << ", val = " << val << ", quant = " << quant << ", clust + quant = " << clust + quant << std::endl;
                    std::cout << "  fine = " << fine[j + idx * dim] << ", diff = " << src - clust << std::endl;
                }
                error += diff * diff;
                coarse_err += (clust - src) * (clust - src);
            }
            total_err += error;
        }
        std::cout << "avg error: " << total_err / vectors->length << std::endl;
        std::cout << "avg error coarse: " << coarse_err / vectors->length << std::endl;
    }
    ret->bias = std::move(force_bias);
    ret->is_l2 = is_l2;
    return ret;
}

void compute_ann_pq(RawVectorData *vectors, int k, float *query, int *result, Index *raw_index) {
    PQIndex *idx = (PQIndex *) raw_index;
    auto iprods = std::make_unique<float[]>(idx->num_clusters);
    memcpy(iprods.get(), idx->bias.get(), idx->num_clusters * sizeof(float));
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        idx->num_clusters,
        idx->dim,
        1.0f,
        idx->clusters.get(),
        idx->dim,
        query,
        1,
        1.0f,
        iprods.get(),
        1
    );
    auto cluster_indices = std::make_unique<int[]>(idx->num_clusters);
    auto ci_begin = &cluster_indices[0];
    auto ci_end = &cluster_indices[idx->num_clusters];
    std::iota(ci_begin, ci_end, 0);
    std::partial_sort(ci_begin, ci_begin + idx->window, ci_end,
                      [&iprods](int a, int b) {
                          return iprods[a] > iprods[b];
                      });
    int subcodebook_size = 1 << idx->fine_bits;
    int total_cb_size = idx->num_groups * subcodebook_size;
    auto cb_iprods = std::make_unique<float[]>(total_cb_size);
    if (idx->is_l2) {
        memcpy(cb_iprods.get(), idx->codebooks_bias.get(), total_cb_size * sizeof(float));
    }
    int group_dim = (idx->dim + idx->num_groups - 1) / idx->num_groups;
    for (int i = 0; i < idx->num_groups; i++) {
        int start_dim = i * group_dim;
        int cur_dim = std::min(group_dim, idx->dim - start_dim);
        cblas_sgemv(
            CblasRowMajor,
            CblasNoTrans,
            subcodebook_size,
            cur_dim,
            1.0f,
            idx->get_codebook(i),
            cur_dim,
            query + start_dim,
            1,
            idx->is_l2 ? 1.0f : 0.0f,
            &cb_iprods[subcodebook_size * i],
            1
        );
    }
    auto window_results = std::make_unique<WindowResult[]>(vectors->length);
    int cur = 0;
    for (int i = 0; i < idx->window; i++) {
        int c = cluster_indices[i];
        search_cluster(c, &window_results[cur], iprods[c], cb_iprods.get(), idx);
        cur += idx->cluster_start[c + 1] - idx->cluster_start[c];
    }
    auto wr_begin = &window_results[0];
    auto wr_end = &window_results[cur];
    std::partial_sort(wr_begin, wr_begin + k, wr_end);
    for (int i = 0; i < k; i++) {
        result[i] = window_results[i].index;
    }
}

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

template<bool LOG>
void compute_k_means(int num_clusters, int dim, int stride, int num_vectors, float *vectors, float *bias, float *out_clusters, box<int[]> *out_assignments) {
    std::cout << "k-means(" << num_clusters << ", " << dim << "@" << stride << ", x" << num_vectors << ")\n";
    std::unique_ptr<int[]> assignments = std::make_unique<int[]>(num_vectors);
    float *clusters = out_clusters;
    std::unique_ptr<float[]> clusters_temp_box = std::make_unique<float[]>(num_clusters * dim);
    float *clusters_temp = clusters_temp_box.get();
    std::unique_ptr<float[]> clusters_bias = std::make_unique<float[]>(num_clusters);
    std::unique_ptr<int[]> clusters_size = std::make_unique<int[]>(num_clusters);
    std::unique_ptr<float[]> iprods = std::make_unique<float[]>(num_clusters * BLOCK_SIZE);
    std::mt19937 rng;
    std::uniform_int_distribution rand_vec(0, num_vectors - 1);
    rng.seed(0xdeadbeef);

    for (int i = 0; i < num_vectors; i++) {
        assignments[i] = i;
    }
    std::shuffle(assignments.get(), &assignments[num_vectors], rng);

    for (int i = 0; i < num_clusters; i++) {
        int j = assignments[i];
        std::memcpy(&clusters[i * dim],
                    &vectors[j * stride],
                    sizeof(float) * dim);
        if (bias) clusters_bias[i] = bias[j];
    }

    int iteration = 0;
    bool converged = false;
    while (!converged) {
        using std::ios;
        ScopedTimer<LOG> timer("k-means", ++iteration);
        if constexpr (LOG) std::cout << "- iteration " << iteration << std::endl;
        int changed = 0;
        float energy = 0.0f;
        compute_assignments(
            num_vectors,
            dim,
            stride,
            vectors,
            bias,
            num_clusters,
            clusters,
            nullptr,
            clusters_bias.get(),
            iprods.get(), // scratch
            assignments.get(),
            &energy,
            &changed
        );
        memset(clusters_size.get(), 0, sizeof(int) * num_clusters);
        memset(clusters_temp, 0, sizeof(float) * num_clusters * dim);
        {
            // update centroids
            for (int i = 0; i < num_vectors; i++) {
                int assignment = assignments[i];
                cblas_saxpy(
                    dim,
                    1.0f,
                    &vectors[i * stride],
                    1,
                    &clusters_temp[assignment * dim],
                    1
                );
                clusters_size[assignment]++;
            }
        }
        // rescale cluster center
        for (int i = 0; i < num_clusters; i++) {
            if (clusters_size[i] == 0) {
                // int j = rand_vec(rng);
                // std::memcpy(&clusters_temp[i * dim],
                //             &vectors[j * stride],
                //             sizeof(float) * dim);
                // if constexpr (LOG) std::cout << "- resetting!" << std::endl;
                continue;
            }
            cblas_sscal(
                dim,
                1.0f / clusters_size[i],
                &clusters_temp[i * dim],
                1
            );
        }
        std::swap(clusters, clusters_temp);
        int nsplit = 0;
        for (int i = 0; i < num_clusters; i++) {
            if (clusters_size[i] != 0) {
                continue;
            }
            int o;
            for (o = 0; true; o = (o+1) % num_clusters) {
                float p = (clusters_size[o] - 1.0f) / (num_vectors - num_clusters);
                float r = std::generate_canonical<float, 16>(rng);
                if (r < p) break;
            }
            memcpy(&clusters[i * dim], &clusters[o * dim], sizeof(float) * dim);
            for (int k = 0; k < dim; k++) {
                if (k % 2) {
                    clusters[i * dim + k] *= 1.0 + 1.0 / 1024;
                    clusters[o * dim + k] *= 1.0 - 1.0 / 1024;
                } else {
                    clusters[i * dim + k] *= 1.0 - 1.0 / 1024;
                    clusters[o * dim + k] *= 1.0 + 1.0 / 1024;
                }
            }
            clusters_size[i] = clusters_size[o] / 2;
            clusters_size[o] -= clusters_size[i];
            nsplit++;
        }
        converged = changed == 0;
        if constexpr (LOG) std::cout << "- " << changed << " assignments changed" << std::endl;
        if constexpr (LOG) std::cout << "- energy: " << energy << std::endl;
        if constexpr (LOG) if (nsplit) std::cout << "- split: " << nsplit << std::endl;
        if (changed < num_vectors / 500 && iteration >= 20) {
            // std::cout << "- stopping early!" << std::endl;
            // break;
        }
        // update clusters_bias if bias != nullptr
        if (bias) {
            compute_l2_bias(dim, num_clusters, clusters, dim, clusters_bias.get());
        }
    }
    if (out_assignments) {
        *out_assignments = std::move(assignments);
    }
}

void compute_assignments(
    int num_vectors,
    int dim,
    int stride,
    float *vectors,
    float *bias,
    int num_clusters,
    float *clusters,
    int *clusters_size,
    float *clusters_bias,
    float *iprods, // scratch
    int *assignments,
    float *energy,
    int *changed
) {
    if (clusters_size) {
        memset(clusters_size, 0, sizeof(int) * num_clusters);
    }
    for (int i = 0; i < num_vectors; i += BLOCK_SIZE) {
        if (bias) {
            for (int j = i; j < i + BLOCK_SIZE && j < num_vectors; j++) {
                std::memcpy(&iprods[(j - i) * num_clusters],
                            clusters_bias,
                            num_clusters * sizeof(float));
            }
        }
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                std::min(BLOCK_SIZE, num_vectors - i),
                num_clusters,
                dim,
                1.0,
                &vectors[i * stride],
                stride,
                clusters,
                dim,
                bias ? 1.0 : 0.0,
                iprods,
                num_clusters
            );
        }
        // compute assignments, update centroids
        for (int j = i; j < i + BLOCK_SIZE && j < num_vectors; j++) {
            int assignment = simd_argmax(
                &iprods[(j - i) * num_clusters],
                num_clusters
            );
            if (energy) {
                // L2 dist = ||p||^2 + ||q||^2 - 2 <p, q>
                float dist = bias[j] + iprods[(j - i) * num_clusters + assignment];
                dist *= -2.0;
                *energy += dist;
            }
            if (changed && assignment != assignments[j]) {
                (*changed)++;
            }
            if (clusters_size) {
                clusters_size[assignment]++;
            }
            assignments[j] = assignment;
        }
    }
}

box<float[]> transform_fine_vectors(RawVectorData *vectors, int num_clusters, float *clusters, int *assignments, float *transform) {
    int num_vectors = vectors->length;
    int dim = vectors->dim;
    box<float[]> ret = std::make_unique<float[]>(num_vectors * dim);
    for (int i = 0; i < num_vectors; i++) {
        int j = assignments[i];
        std::memcpy(&ret[i * dim], &clusters[j * dim], sizeof(float) * dim);
    }
    cblas_sgemm(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        dim,
        num_vectors,
        dim,
        1.0f,
        transform,
        dim,
        vectors->vec + 1,
        dim + 1,
        -1.0f,
        &ret[0],
        dim
    );
    return ret;
}

void generate_transform(float *mat, int dim) {
    // identity for now...
    memset(mat, 0, sizeof(float) * dim * dim);
    for (int i = 0; i < dim; i++) {
        mat[i * dim + i] = 1.0f;
    }
}

template<int G>
void write_bytes(char *out, int x) {
    memcpy(out, &x, G);
}

template<int G>
int read_bytes(char *in) {
    int x = 0;
    memcpy(&x, in, G);
    return x;
}

template<int G>
void write_helper(int num_vectors, int *assignments, int *scatter, int qvec_size, int offset, char *out) {
    for (int i = 0; i < num_vectors; i++) {
        write_bytes<G>(&out[qvec_size * scatter[i] + G * offset], assignments[i]);
    }
}

void write_assignments(int num_vectors, int *assignments, int *scatter, int bits, int num_groups, int dim_i, char *out) {
    int group_size = byte_size(bits);
    int qvec_size = roundup_line(group_size * num_groups);
    if (group_size == 1) {
        write_helper<1>(num_vectors, assignments, scatter, qvec_size, dim_i, out);
    } else if (group_size == 2) {
        write_helper<2>(num_vectors, assignments, scatter, qvec_size, dim_i, out);
    } else if (group_size == 4) { [[unlikely]]
        write_helper<4>(num_vectors, assignments, scatter, qvec_size, dim_i, out);
    } else if (group_size == 3) { [[unlikely]]
        write_helper<3>(num_vectors, assignments, scatter, qvec_size, dim_i, out);
    } else { [[unlikely]]
        throw std::runtime_error("invalid number of bytes to write");
    }
}

template<int G>
void search_helper(int start, int N, WindowResult *out, int qvec_size, float cluster_iprod, float *iprods, PQIndex *idx) {
    int subcodebook_size = 1 << idx->fine_bits;
    for (int i = 0; i < N; i++) {
        out[i].index = idx->cluster_values[start + i];
        float iprod = cluster_iprod;
        for (int j = 0; j < idx->num_groups; j++) {
            int position = qvec_size * (start + i) + G * j;
            int q = read_bytes<G>(&idx->clustered_quant[position]);
            iprod += iprods[j * subcodebook_size + q];
        }
        out[i].iprod = iprod;
    }
}

static void search_cluster(int c, WindowResult *out, float cluster_iprod, float *iprods, PQIndex *idx) {
    int start = idx->cluster_start[c];
    int N = idx->cluster_start[c + 1] - idx->cluster_start[c];
    int group_size = byte_size(idx->fine_bits);
    int qvec_size = roundup_line(group_size * idx->num_groups);
    if (group_size == 1) {
        search_helper<1>(start, N, out, qvec_size, cluster_iprod, iprods, idx);
    } else if (group_size == 2) {
        search_helper<2>(start, N, out, qvec_size, cluster_iprod, iprods, idx);
    } else if (group_size == 4) { [[unlikely]]
        search_helper<4>(start, N, out, qvec_size, cluster_iprod, iprods, idx);
    } else if (group_size == 3) { [[unlikely]]
        search_helper<3>(start, N, out, qvec_size, cluster_iprod, iprods, idx);
    } else { [[unlikely]]
        throw std::runtime_error("invalid number of bytes to write");
    }
}
