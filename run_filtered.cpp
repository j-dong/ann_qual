#include "kernel_utils.h"
#include "graph_utils.h"
#include "load_files.h"
#include "timer.h"
#include "simd_utils.h"
#include "visited_map.h"
#include "magic.h"

#ifndef NOMINMAX
# define NOMINMAX 1
#endif
#include "inc_cblas.h"
#include "control_threading.h"

#include "argparse/argparse.hpp"

#include <cstring>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <memory>
#include <span>
#include <vector>
#include <cassert>

double overhead_total = 0.0;
double compute_total = 0.0;

constexpr int byte_size(int bits) {
    return (bits + 7) / 8;
}

constexpr int roundup_line(int bytes) {
    return (bytes + 63) / 64 * 64;
}

struct aligned_deleter {
    void operator()(char *p) {
        ::operator delete[](p, std::align_val_t(64));
    }
};

struct min_heap : vertex_heap<CompMinDist> {
    using vertex_heap::vertex_heap;
    std::vector<PQElement> &elements() { return c; }
};

struct max_heap : vertex_heap<CompMaxDist> {
    using vertex_heap::vertex_heap;
    std::vector<PQElement> &elements() { return c; }
};

template<size_t N>
consteval size_t array_size(const std::array<uint32_t, N> &) {
    return N;
}

class StatsProvider;

template<typename T>
class StatsWrapper {
    StatsProvider &provider;

    friend class Index;
    StatsWrapper(StatsProvider &provider) : provider(provider) {}

public:
    void write(const T &value);
};

class StatsProvider {
    std::unique_ptr<char[]> data;
    size_t pos;
    size_t len;
    // auto-generated:
    // uint32_t magic: magic number
    // uint32_t offset: offset of the data
    // in header:
    // uint32_t num_fields: number of fields
    // uint32_t fields[num_fields]: field i size (bytes)
    // char description[]: description of the fields, separated by newlines
    std::span<uint32_t const> header;
    std::string_view description;

    template<typename T>
    friend class StatsWrapper;

    template<typename T>
    void write(T value) {
        memcpy(&data[pos], &value, sizeof(T));
        pos += sizeof(T);
    }

public:

    template<typename T>
    void init(size_t count) {
        data = std::unique_ptr<char[]>(new char[count * sizeof(T)]);
        pos = 0;
        len = count * sizeof(T);
        // T should have a static member std::array<uint32_t, N> header
        // and a static member const char *description
        const std::array<uint32_t, array_size(T::header)> &the_header = T::header;
        header = std::span<uint32_t const>(the_header.begin(), the_header.size());
        description = T::description;
    }

    void writeOut(const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        uint32_t offset = (2 + header.size()) * sizeof(uint32_t);
        uint32_t desc_len = (description.size() & ~0x3) + 4;
        offset += desc_len;
        std::vector<uint32_t> temp_header;
        temp_header.resize(header.size() + 2);
        temp_header[0] = 0x54415453;
        temp_header[1] = offset;
        std::copy(header.begin(), header.end(), temp_header.begin() + 2);
        std::vector<char> desc_buf;
        desc_buf.resize(desc_len);
        std::memcpy(desc_buf.data(), description.data(), description.size());
        ofs.write(reinterpret_cast<char *>(temp_header.data()), temp_header.size() * sizeof(uint32_t));
        ofs.write(desc_buf.data(), desc_len);
        ofs.write(data.get(), len);
    }
};

template<typename T>
void StatsWrapper<T>::write(const T &value) { provider.write(value); }

template<typename Derived>
struct StatsInitHelper {
private:
    template<typename T, typename = void>
    struct has_stats : std::false_type {};
    template<typename T>
    struct has_stats<T, std::void_t<typename T::Stats>> : std::true_type {};

    template<typename T>
    static void init_impl(StatsProvider &stats, size_t count, std::true_type) {
        stats.init<typename T::Stats>(count);
    }

    template<typename T>
    static void init_impl(StatsProvider &, size_t, std::false_type) {
        // Do nothing
    }

public:
    static void init(StatsProvider &stats, size_t count) {
        init_impl<Derived>(stats, count, has_stats<Derived>{});
    }

    static void writeOut(StatsProvider &stats, std::string filename) {
        if constexpr(!has_stats<Derived>::value) { return; }
        stats.writeOut(filename);
    }
};

enum class IndexType {
    Naive,
    PQ,
    HNSW,
    Vamana,
};

struct NaiveIndex;

class Index {
public:
    StatsProvider stats;

protected:
    template<typename Derived>
    void run_filter_test_impl(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) {
        StatsInitHelper<Derived>::init(stats, query->length);
        memset(result, -1, query->length * k * sizeof(int));
        ScopedTimer timer("run_filter_test");
        for (int qi = 0; qi < query->length; qi++) {
            static_cast<Derived *>(this)->query_filtered(&query->at(qi, 0), &result[qi * k], k, filter_idx);
            if constexpr (std::is_same_v<Derived, NaiveIndex>) {
                if (qi % 1000 == 0) {
                    std::cout << "progress: " << qi << "/" << query->length << std::endl;
                }
            }
        }
        timer.print_timer_message("avg filtered query latency", -1, timer.get_ms() / query->length);
        if (false) StatsInitHelper<Derived>::writeOut(stats, stats_filename);
    }

    template<typename Derived>
    StatsWrapper<typename Derived::Stats> statsWriter(Derived *) {
        return StatsWrapper<typename Derived::Stats>(stats);
    }

public:
    virtual ~Index() = default;
    virtual IndexType type() const = 0;
    virtual void query_filtered(float *query, int *result, int k, int filter_idx) = 0;
    virtual int get_num_points() const = 0;
    virtual void run_filter_test(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) = 0;
};

struct Vertex {
    int numNeighbors = 0;
    VertexPtr below = nullptr;
    int id;
    float *data;
};

struct GraphIndex {
    int dim;
    int numPoints;
    std::vector<Vertex> vertices;
    std::vector<VertexPtr> all_neighbors;
    std::unique_ptr<VisitedMap> visited_ptr;

    Vertex &get(VertexPtr p) {
        return vertices[p.i];
    }

    // std::span<VertexPtr> neighbors(VertexPtr p) = 0;
};

struct NaiveIndex : Index {
    RawVectorData *vectors;
    std::vector<float> bias;

    IndexType type() const override { return IndexType::Naive; }
    int get_num_points() const override { return vectors->length; }
    void query_filtered(float *query, int *result, int k, int filter_idx) override;
    void run_filter_test(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) override {
        run_filter_test_impl<NaiveIndex>(query, result, k, filter_idx, stats_filename);
    }
};

struct PQIndex : Index {
    int num_points;
    int dim;
    int num_clusters;
    int fine_bits;
    int num_groups;
    int window;
    std::vector<float> clusters;
    std::vector<float> clusters_bias;
    std::vector<float> codebooks;
    std::vector<float> transform;
    std::vector<int> cluster_start;
    std::vector<int> cluster_values;
    std::vector<float> bias;
    std::unique_ptr<char[], aligned_deleter> clustered_quant;
    bool is_l2;

    struct Stats {
        uint32_t clusters_explored;

        static constexpr std::array<uint32_t, 2> header = {
            1, sizeof(uint32_t)
        };
        static constexpr char description[] = "clusters explored\n";
    };

    IndexType type() const override { return IndexType::PQ; }
    int get_num_points() const override { return num_points; }
    void query_filtered(float *query, int *result, int k, int filter_idx) override;
    void run_filter_test(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) override {
        run_filter_test_impl<PQIndex>(query, result, k, filter_idx, stats_filename);
    }
};

struct HNSWIndex : Index, GraphIndex {
    int maxDegree;
    int maxLevel;
    VertexPtr entry;

    struct Stats {
        uint32_t true_visited;
        uint32_t false_visited;

        static constexpr std::array<uint32_t, 3> header = {
            2, sizeof(uint32_t), sizeof(uint32_t)
        };
        static constexpr char description[] = "true visited\nfalse visited\n";
    };

    bool isLayer0(VertexPtr p) {
        return p.i < (uint32_t) numPoints;
    }

    std::span<VertexPtr> neighbors(VertexPtr p) {
        if (isLayer0(p)) {
            return std::span<VertexPtr>(&all_neighbors[2 * maxDegree * p.i],
                                        get(p).numNeighbors);
        }
        int idx = 2 * maxDegree * numPoints
            + maxDegree * (p.i - numPoints);
        return std::span<VertexPtr>(&all_neighbors[idx],
                                    get(p).numNeighbors);
    }

    IndexType type() const override { return IndexType::HNSW; }
    int get_num_points() const override { return numPoints; }
    void query_filtered(float *query, int *result, int k, int filter_idx) override;
    void run_filter_test(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) override {
        run_filter_test_impl<HNSWIndex>(query, result, k, filter_idx, stats_filename);
    }
};

struct VamanaIndex : Index, GraphIndex {
    int maxDegree;
    VertexPtr entry;

    using Stats = HNSWIndex::Stats;

    std::span<VertexPtr> neighbors(VertexPtr p) {
        return std::span<VertexPtr>(&all_neighbors[p.i * maxDegree],
                                    get(p).numNeighbors);
    }

    IndexType type() const override { return IndexType::Vamana; }
    int get_num_points() const override { return numPoints; }
    void query_filtered(float *query, int *result, int k, int filter_idx) override;
    void run_filter_test(RawVectorData *query, int *result, int k, int filter_idx, std::string stats_filename) override {
        run_filter_test_impl<VamanaIndex>(query, result, k, filter_idx, stats_filename);
    }
};

std::unique_ptr<VisitedMap> new_cleared_map(size_t size) {
    auto ret = std::make_unique<VisitedMap>(size);
    memset(ret->vec, 0, size);
    ret->cleared = true;
    ret->tag = 1;
    return ret;
}

// loading

struct PQHeader {
    int magic;
    int dim;
    int num_clusters;
    int fine_bits;
    int num_groups;
    int num_points;
};

struct HNSWHeader {
    int magic;
    int dim;
    int max_degree;
    int num_points;
    int num_vertices;
    int max_level;
    int entry;
};

struct VamanaHeader {
    int magic;
    int dim;
    int max_degree;
    int num_points;
    int entry;
};

std::unique_ptr<NaiveIndex> load_naive(RawVectorData *raw_data) {
    // just precalculate the bias
    auto index = std::make_unique<NaiveIndex>();
    index->vectors = raw_data;
    index->bias.resize(index->vectors->length);
    compute_l2_bias(raw_data->dim, raw_data->length, &raw_data->at(0, 0), raw_data->dim + 1, index->bias.data());
    return index;
}

std::unique_ptr<PQIndex> load_pq(std::istream &is, [[maybe_unused]] RawVectorData *raw_data) {
    PQHeader header;
    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (header.magic != magic::PQ_L2 && header.magic != magic::PQ_IP) {
        throw std::runtime_error("Invalid magic number");
    }
    auto index = std::make_unique<PQIndex>();
    index->is_l2 = (header.magic == magic::PQ_L2);
    index->dim = header.dim;
    index->num_clusters = header.num_clusters;
    index->fine_bits = header.fine_bits;
    index->num_groups = header.num_groups;
    index->num_points = header.num_points;

    // resize vectors
    int group_dim = (index->dim + index->num_groups - 1) / index->num_groups;
    index->clusters.resize(header.num_clusters * header.dim);
    index->clusters_bias.resize(header.num_clusters);
    index->codebooks.resize(header.num_groups * (1 << header.fine_bits) * group_dim);
    index->transform.resize(header.dim * header.dim);
    index->cluster_start.resize(header.num_clusters + 1);
    index->cluster_values.resize(header.num_points);
    index->bias.resize(header.num_points);
    int qvec_size = roundup_line(header.num_groups * byte_size(header.fine_bits));
    index->clustered_quant = std::unique_ptr<char[], aligned_deleter>(
            new (std::align_val_t(64)) char[qvec_size * header.num_points]
        );

    // read vectors
    is.read(reinterpret_cast<char *>(index->clusters.data()), index->clusters.size() * sizeof(float));
    is.read(reinterpret_cast<char *>(index->clusters_bias.data()), index->clusters_bias.size() * sizeof(float));
    is.read(reinterpret_cast<char *>(index->codebooks.data()), index->codebooks.size() * sizeof(float));
    is.read(reinterpret_cast<char *>(index->transform.data()), index->transform.size() * sizeof(float));
    is.read(reinterpret_cast<char *>(index->cluster_start.data()), index->cluster_start.size() * sizeof(int));
    is.read(reinterpret_cast<char *>(index->cluster_values.data()), index->cluster_values.size() * sizeof(int));
    is.read(reinterpret_cast<char *>(index->bias.data()), index->bias.size() * sizeof(float));
    is.read(reinterpret_cast<char *>(index->clustered_quant.get()), qvec_size * header.num_points);
    return index;
}

std::unique_ptr<HNSWIndex> load_hnsw(std::istream &is, RawVectorData *raw_data) {
    HNSWHeader header;
    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (header.magic != magic::HNSW) {
        throw std::runtime_error("Invalid magic number");
    }
    auto index = std::make_unique<HNSWIndex>();
    index->dim = header.dim;
    index->numPoints = header.num_points;
    index->maxDegree = header.max_degree;
    index->maxLevel = header.max_level;
    index->entry = VertexPtr(header.entry);

    // same as Vertex, but for consistency
    struct HNSWVertex {
        int numNeighbors;
        VertexPtr below;
        int id;
        float *data;
    };
    std::vector<HNSWVertex> temp_vertices;
    temp_vertices.resize(header.num_vertices);
    index->vertices.resize(header.num_vertices);
    is.read(reinterpret_cast<char *>(temp_vertices.data()), temp_vertices.size() * sizeof(HNSWVertex));
    for (int i = 0; i < header.num_vertices; i++) {
        index->vertices[i].numNeighbors = temp_vertices[i].numNeighbors;
        index->vertices[i].below = temp_vertices[i].below;
        index->vertices[i].id = temp_vertices[i].id;
        index->vertices[i].data = &raw_data->at(temp_vertices[i].id, 0);
    }

    index->all_neighbors.resize(header.num_points * header.max_degree + header.num_vertices * header.max_degree);
    memset(index->all_neighbors.data(), -1, index->all_neighbors.size() * sizeof(VertexPtr));
    std::vector<VertexPtr> vertex_neighbors;
    vertex_neighbors.resize(header.max_degree * 2);
    for (int i = 0; i < header.num_vertices; i++) {
        is.read(reinterpret_cast<char *>(vertex_neighbors.data()), index->vertices[i].numNeighbors * sizeof(VertexPtr));
        auto in = index->neighbors(VertexPtr(i));
        std::copy_n(vertex_neighbors.begin(), index->vertices[i].numNeighbors, in.begin());
    }

    index->visited_ptr = new_cleared_map(index->vertices.size());
    return index;
}

std::unique_ptr<VamanaIndex> load_vamana(std::istream &is, RawVectorData *raw_data) {
    VamanaHeader header;
    is.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (header.magic != magic::VAMANA) {
        throw std::runtime_error("Invalid magic number");
    }
    auto index = std::make_unique<VamanaIndex>();
    index->dim = header.dim;
    index->numPoints = header.num_points;
    index->maxDegree = header.max_degree;
    index->entry = VertexPtr(header.entry);

    struct VamanaVertex {
        int numNeighbors = 0;
        float *data;
    };
    std::vector<VamanaVertex> temp_vertices;
    temp_vertices.resize(header.num_points);
    index->vertices.resize(header.num_points);
    is.read(reinterpret_cast<char *>(temp_vertices.data()), temp_vertices.size() * sizeof(VamanaVertex));
    for (int i = 0; i < header.num_points; i++) {
        index->vertices[i].numNeighbors = temp_vertices[i].numNeighbors;
        index->vertices[i].data = &raw_data->at(i, 0);
        index->vertices[i].id = i;
    }

    index->all_neighbors.resize(header.num_points * header.max_degree);
    memset(index->all_neighbors.data(), -1, index->all_neighbors.size() * sizeof(VertexPtr));
    std::vector<VertexPtr> vertex_neighbors;
    vertex_neighbors.resize(header.max_degree);
    for (int i = 0; i < header.num_points; i++) {
        is.read(reinterpret_cast<char *>(vertex_neighbors.data()), index->vertices[i].numNeighbors * sizeof(VertexPtr));
        auto in = index->neighbors(VertexPtr(i));
        std::copy_n(vertex_neighbors.begin(), index->vertices[i].numNeighbors, in.begin());
    }

    index->visited_ptr = new_cleared_map(index->vertices.size());
    return index;
}

std::unique_ptr<Index> load_index(const std::string &path, RawVectorData *raw_data) {
    if (path == "naive") {
        return load_naive(raw_data);
    }

    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open file");
    }
    int magic;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    is.seekg(0, std::ios::beg); // reset to beginning of file
    if (magic == magic::PQ_L2 || magic == magic::PQ_IP) {
        return load_pq(is, raw_data);
    } else if (magic == magic::HNSW) {
        return load_hnsw(is, raw_data);
    } else if (magic == magic::VAMANA) {
        return load_vamana(is, raw_data);
    }
    throw std::runtime_error("Unknown magic number");
}

// main function

int main(int argc, char **argv) {
    argparse::ArgumentParser parser("run_filtered");
    parser.add_argument("index_path")
        .help("path to index file")
        .required();
    parser.add_argument("-k", "--num-results")
        .help("number of vectors to search for")
        .default_value(30)
        .scan<'d', int>();
    parser.add_argument("-s", "--selectivity")
        .help("filter selectivity")
        .default_value(0.01)
        .scan<'f', float>();
    parser.add_argument("-p", "--path")
        .help("path prefix for vector data; required files are PREFIX_base.fvecs, PREFIX_learn.fvecs, PREFIX_query.fvecs, PREFIX_groundtruth.ivecs")
        .default_value("G:\\vectors\\siftsmall\\siftsmall");
    parser.parse_args(argc, argv);

    load_files(parser.get<std::string>("-p"));

    auto index = load_index(parser.get<std::string>("index_path"), &data_base);

    std::string fname = parser.get<std::string>("index_path");
    if (fname.find("saved_") != std::string::npos) {
        fname = fname.substr(6);
    }

    std::stringstream ss;
    ss << "stats_filtered_" << fname << "_" << std::fixed << std::setprecision(3) << parser.get<float>("-s");

    std::vector<int> result;
    result.resize(parser.get<int>("-k") * data_query.length);
    index->run_filter_test(&data_query, result.data(), parser.get<int>("-k"), index->get_num_points() - (int) (parser.get<float>("-s") * index->get_num_points()), ss.str());

    if (false) {
        // index fname is something like saved_out_pq_k1024_b8_m32_w16_full
        // --> filtered_out_pq_k1024_b8_m32_w16_full
        ss.str(std::string()); ss.clear();
        ss << "filtered_" << fname << "_" << std::fixed << std::setprecision(3) << parser.get<float>("-s");
        std::ofstream ofs(ss.str(), std::ios::binary);
        ofs.write(reinterpret_cast<char *>(result.data()), result.size() * sizeof(int));
        std::cout << "filtered results saved to " << ss.str() << std::endl;
    }

    std::cout << "overhead: " << overhead_total / data_query.length << " ms" << std::endl;
    std::cout << "compute:  " << compute_total / data_query.length << " ms" << std::endl;

    close_files();
    return 0;
}

// filtering

struct SortedResult {
    int i;
    float v;
};

struct CompDesc {
    bool operator()(const SortedResult &a, const SortedResult &b) {
        return a.v > b.v;
    }
};

//////////////////// NAIVE                                  ////////////////////
// MARK: naive query impl

void NaiveIndex::query_filtered(float *query, int *result, int k, int filter_idx) {
    // pre-filtering
    int dim = vectors->dim;

    std::vector<SortedResult> results;
    results.reserve(vectors->length);

    // pretend we don't know the number of filtered points
    results.clear();
    for (int i = 0; i < vectors->length; i++) {
        if (i >= filter_idx) {
            results.push_back(SortedResult{i, 0});
        }
    }

    int num_filtered = (int) results.size();

    // calculate iprods
    std::vector<float> iprods(num_filtered);
#pragma omp parallel for
    for (int i = 0; i < num_filtered; i++) {
        iprods[i] = cblas_sdot(dim, &vectors->at(results[i].i, 0), 1, query, 1) + bias[results[i].i];
    }
    // sort
    if (k < num_filtered) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(), CompDesc());
    } else {
        std::sort(results.begin(), results.end(), CompDesc());
    }

    for (int i = 0; i < std::min(k, (int) results.size()); i++) {
        result[i] = results[i].i;
    }
}


//////////////////// PQ                                   ////////////////////
// MARK: pq query impl

static int search_cluster_filtered(int c, SortedResult *out, float cluster_iprod, float *iprods, PQIndex *idx, int filter_idx);

void PQIndex::query_filtered(float *query, int *result, int k, int filter_idx) {
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    start = std::chrono::steady_clock::now();
    // Compute cluster inner products
    std::vector<float> iprods(num_clusters);
    std::vector<float> query_xformed(dim);
    memcpy(iprods.data(), clusters_bias.data(), num_clusters * sizeof(float));
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        num_clusters,
        dim,
        1.0f,
        clusters.data(),
        dim,
        query,
        1,
        1.0f,
        iprods.data(),
        1
    );

    // Sort clusters by inner product
    std::vector<int> cluster_indices(num_clusters);
    std::iota(cluster_indices.begin(), cluster_indices.end(), 0);
    std::sort(cluster_indices.begin(), cluster_indices.end(),
              [&iprods](int a, int b) {
                  return iprods[a] > iprods[b];
              });

    // Transform query for fine quantizer
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        dim,
        dim,
        1.0,
        transform.data(),
        dim,
        query,
        1,
        0.0,
        query_xformed.data(),
        1
    );

    // Compute codebook inner products
    int subcodebook_size = 1 << fine_bits;
    int total_cb_size = num_groups * subcodebook_size;
    std::vector<float> cb_iprods(total_cb_size);
    int group_dim = (dim + num_groups - 1) / num_groups;
    for (int i = 0; i < num_groups; i++) {
        int start_dim = i * group_dim;
        int cur_dim = std::min(group_dim, dim - start_dim);
        for (int row = 0; row < subcodebook_size; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < cur_dim; ++col) {
                float a = codebooks[i * subcodebook_size * group_dim + row * cur_dim + col];
                float x = query_xformed[start_dim + col];
                sum += a * x;
            }
            cb_iprods[subcodebook_size * i + row] = sum;
        }
    }

    // Search clusters with dynamic window size
    std::vector<SortedResult> results;
    results.reserve(num_points);
    int cur = 0;
    int filtered_found = 0;
    int cluster_idx = 0;

    end = std::chrono::steady_clock::now();
    overhead_total += std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>
    >(end - start).count();
    start = std::chrono::steady_clock::now();
    while (filtered_found < k && cluster_idx < num_clusters) {
        int c = cluster_indices[cluster_idx];
        float iprod = iprods[c] - clusters_bias[c];
        int cluster_size = cluster_start[c + 1] - cluster_start[c];
        results.resize(cur + cluster_size);
        filtered_found += search_cluster_filtered(c, &results[cur], iprod, cb_iprods.data(), this, filter_idx);
        cur += cluster_size;
        cluster_idx++;
    }
    // keep going for window=64 more
    int window_end = std::min(cur + 64, num_clusters);
    while (cluster_idx < window_end) {
        int c = cluster_indices[cluster_idx];
        float iprod = iprods[c] - clusters_bias[c];
        int cluster_size = cluster_start[c + 1] - cluster_start[c];
        results.resize(cur + cluster_size);
        filtered_found += search_cluster_filtered(c, &results[cur], iprod, cb_iprods.data(), this, filter_idx);
        cur += cluster_size;
        cluster_idx++;
    }
    statsWriter(this).write(Stats{.clusters_explored = (uint32_t) cluster_idx});

    end = std::chrono::steady_clock::now();
    compute_total += std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>
    >(end - start).count();

    // same as naive
    if (k < filtered_found) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(), CompDesc());
    } else {
        std::sort(results.begin(), results.end(), CompDesc());
    }

    for (int i = 0; i < std::min(k, (int) results.size()); i++) {
        result[i] = results[i].i;
    }
}

template<int G>
int read_bytes(char *in) {
    int x = 0;
    memcpy(&x, in, G);
    return x;
}

template<int G>
void search_helper_filtered(int start, int N, SortedResult *out, int qvec_size, float cluster_iprod, float *iprods, PQIndex *idx, int filter_idx, int &filtered_count) {
    int subcodebook_size = 1 << idx->fine_bits;
    filtered_count = 0;
    for (int i = 0; i < N; i++) {
        int index = idx->cluster_values[start + i];
        if (index < filter_idx) continue;
        float iprod = cluster_iprod;
        for (int j = 0; j < idx->num_groups; j++) {
            int position = qvec_size * (start + i) + G * j;
            int q = read_bytes<G>(&idx->clustered_quant[position]);
            iprod += iprods[j * subcodebook_size + q];
        }
        out[filtered_count].i = index;
        out[filtered_count].v = iprod + idx->bias[start + i];
        filtered_count++;
    }
}

static int search_cluster_filtered(int c, SortedResult *out, float cluster_iprod, float *iprods, PQIndex *idx, int filter_idx) {
    int start = idx->cluster_start[c];
    int N = idx->cluster_start[c + 1] - idx->cluster_start[c];
    int group_size = byte_size(idx->fine_bits);
    int qvec_size = roundup_line(group_size * idx->num_groups);
    int filtered_count = 0;
    if (group_size == 1) {
        search_helper_filtered<1>(start, N, out, qvec_size, cluster_iprod, iprods, idx, filter_idx, filtered_count);
    } else if (group_size == 2) {
        search_helper_filtered<2>(start, N, out, qvec_size, cluster_iprod, iprods, idx, filter_idx, filtered_count);
    } else [[unlikely]] if (group_size == 4) {
        search_helper_filtered<4>(start, N, out, qvec_size, cluster_iprod, iprods, idx, filter_idx, filtered_count);
    } else if (group_size == 3) {
        search_helper_filtered<3>(start, N, out, qvec_size, cluster_iprod, iprods, idx, filter_idx, filtered_count);
    } else {
        throw std::runtime_error("invalid number of bytes to read");
    }
    return filtered_count;
}

//////////////////// GRAPH UTILS                            ////////////////////
// MARK: graph utils

template<typename TheIndex>
max_heap hnswSearchLayer(VisitedMap &visited, HNSWIndex::Stats &stats, TheIndex &graph, int filter_idx, float *data, PQElement ep, int ef) {
    min_heap candidates_true;
    min_heap candidates_false;
    max_heap nearest;
    float farthest_dist = std::numeric_limits<float>::infinity();

    if (graph.get(ep.vertex).id < filter_idx) {
        candidates_false.push(ep);
    } else {
        candidates_true.push(ep);
        nearest.push(ep);
        farthest_dist = ep.dist;
    }

    while (!candidates_true.empty() || !candidates_false.empty()) {
        PQElement cur(0, nullptr);
        if (!candidates_true.empty()) {
            cur = candidates_true.top();
            candidates_true.pop();
            stats.true_visited++;
        } else {
            cur = candidates_false.top();
            candidates_false.pop();
            stats.false_visited++;
        }

        if (cur.dist > farthest_dist) {
            break;
        }

        for (VertexPtr e : graph.neighbors(cur.vertex)) {
            assert(e);
            int id = graph.get(e).id;
            assert(id >= 0 && id < graph.get_num_points());
            if (visited[id]) continue;
            visited.set(id);
            float dist = simd_l2dist(graph.get(e).data, data, graph.dim);
            if (dist < farthest_dist || nearest.size() < (size_t) ef) {
                if (id < filter_idx) {
                    candidates_false.push(PQElement(dist, e));
                    continue;
                }
                candidates_true.push(PQElement(dist, e));
                if (nearest.size() >= (size_t) ef) {
                    if (dist < farthest_dist) {
                        nearest.pop();
                        nearest.push(PQElement(dist, e));
                        farthest_dist = nearest.top().dist;
                    }
                } else {
                    nearest.push(PQElement(dist, e));
                    farthest_dist = nearest.top().dist;
                }
            }
        }
    }

    visited.reset();

    return nearest;
}

template<typename TheIndex>
PQElement hnswSearchLayer1(VisitedMap &visited, HNSWIndex::Stats &stats, TheIndex &graph, int filter_idx, float *data, PQElement ep) {
    min_heap candidates_true;
    min_heap candidates_false;
    PQElement nearest = ep;
    bool nearest_ok;

    if (graph.get(ep.vertex).id < filter_idx) {
        candidates_false.push(ep);
        nearest_ok = false;
    } else {
        candidates_true.push(ep);
        nearest_ok = true;
    }

    while (!candidates_true.empty() || !candidates_false.empty()) {
        PQElement cur(0, nullptr);
        if (!candidates_true.empty()) {
            cur = candidates_true.top();
            candidates_true.pop();
            stats.true_visited++;
        } else {
            cur = candidates_false.top();
            candidates_false.pop();
            stats.false_visited++;
        }

        if (nearest_ok && cur.dist > nearest.dist) {
            break;
        }

        for (VertexPtr e : graph.neighbors(cur.vertex)) {
            assert(e);
            int id = graph.get(e).id;
            if (visited[id]) continue;
            visited.set(id);
            float dist = simd_l2dist(graph.get(e).data, data, graph.dim);
            bool better = dist < nearest.dist;
            bool filter_ok = id >= filter_idx;
            if (filter_ok && !nearest_ok) better = true;
            if (better) {
                if (!filter_ok) {
                    candidates_false.push(PQElement(dist, e));
                    nearest = PQElement(dist, e);
                } else {
                    candidates_true.push(PQElement(dist, e));
                    nearest = PQElement(dist, e);
                    nearest_ok = true;
                }
            }
        }
    }

    visited.reset();

    return nearest;
}

//////////////////// HNSW                                   ////////////////////
// MARK: hnsw query impl

void HNSWIndex::query_filtered(float *query, int *result, int k, int filter_idx) {
    VisitedMap &visited = *visited_ptr;
    Stats stats = {};
    PQElement ep(simd_l2dist(get(entry).data, query, dim), entry);  
    for (int cur_level = maxLevel; cur_level > 0; cur_level--) {
        ep = hnswSearchLayer1(visited, stats, *this, filter_idx, query, ep);
        ep.vertex = get(ep.vertex).below;
    }
    max_heap nearest = hnswSearchLayer<HNSWIndex>(visited, stats, *this, filter_idx, query, ep, k);
    statsWriter(this).write(stats);
    while (nearest.size() > (size_t) k) {
        nearest.pop();
    }
    std::vector<PQElement> ret = std::move(nearest).move_elements();
    static_assert(std::is_same<decltype(nearest)::value_compare, CompareDistance>::value, "must sort using same comparator");
    std::sort_heap(ret.begin(), ret.end(), CompareDistance());
    for (int i = 0; i < std::min(k, (int) ret.size()); i++) {
        result[i] = ret[i].vertex.i;
    }
}

//////////////////// VAMANA                                  ////////////////////
// MARK: vamana query impl


void VamanaIndex::query_filtered(float *query, int *result, int k, int filter_idx) {
    VisitedMap &visited = *visited_ptr;
    Stats stats = {};
    PQElement ep(simd_l2dist(get(entry).data, query, dim), entry);
    max_heap nearest = hnswSearchLayer<VamanaIndex>(visited, stats, *this, filter_idx, query, ep, k);
    statsWriter(this).write(stats);
    while (nearest.size() > (size_t) k) {
        nearest.pop();
    }
    std::vector<PQElement> ret = std::move(nearest).move_elements();
    static_assert(std::is_same<decltype(nearest)::value_compare, CompareDistance>::value, "must sort using same comparator");
    std::sort_heap(ret.begin(), ret.end(), CompareDistance());
    for (int i = 0; i < std::min(k, (int) ret.size()); i++) {
        result[i] = ret[i].vertex.i;
    }
}
