#include "kernel_hnsw.h"
#include "kernel_utils.h"
#include "load_files.h"
#include "timer.h"
#include "simd_utils.h"
#include "graph_utils.h"
#include "visited_map.h"
#include <type_traits>

#ifndef NOMINMAX
# define NOMINMAX 1
#endif

#include <cstring>
#include <vector>
#include <queue>
#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <span>

#ifdef _OPENMP
# include <mutex>
# include <shared_mutex>
# include <atomic>
# include <omp.h>
# define WITH_SHARED_LOCK(mutex) std::shared_lock my_lock(mutex)
# define WITH_SHARED_LOCK_2(mutex) std::shared_lock my_lock_2(mutex)
# define WITH_UNIQUE_LOCK(mutex) std::unique_lock my_lock(mutex)
# define WITH_UNIQUE_LOCK_2(mutex) std::unique_lock my_lock_2(mutex)
# define DEFAULT_THREADED true
# define UNIQUE_LOCK(Ty) std::unique_lock<Ty>
# define UNIQUE_LOCK_TY(Ty) std::unique_lock<Ty>
# define SHARED_LOCK_TY(Ty) std::unique_lock<Ty>
# define COND_LOCK my_lock
#else
# define WITH_SHARED_LOCK(mutex)
# define WITH_UNIQUE_LOCK(mutex)
# define DEFAULT_THREADED false
# define UNIQUE_LOCK_TY(Ty) void
# define UNIQUE_LOCK(Ty) UNIQUE_LOCK_M
# define SHARED_LOCK_TY(Ty) UNIQUE_LOCK_M
# define UNIQUE_LOCK_M(val) 0
# define COND_LOCK(val) (*my_lock)()
namespace {
    template<typename T>
    struct UNIQUE_LOCK_M {};
}
#endif

#include "argparse/argparse.hpp"

struct HNSWIndex;

namespace {
struct Vertex {
    int numNeighbors = 0;
    VertexPtr below = nullptr;
    int id;
    float *data;

    Vertex(int id, float *data) : id(id), data(data) {}
};

enum class Lowering {
    SameLevel,
    Lower,
};

template<typename Comp>
struct my_heap : vertex_heap<Comp> {
    template<typename Comp2>
    friend struct my_heap;

    const std::vector<PQElement> &elements() { return this->c; }
    const std::vector<PQElement> move_elements() && { return std::move(this->c); }

    my_heap() : vertex_heap<Comp>() {}
    explicit my_heap(std::vector<PQElement> &&vec) : vertex_heap<Comp>(std::move(vec)) {}
    template<typename Comp2>
    explicit my_heap(const my_heap<Comp2> &other) : vertex_heap<Comp>(other) {}

    std::vector<PQElement> &elements_for_lower() { return this->c; }
};

struct max_heap : my_heap<CompMaxDist> {
    using my_heap<CompMaxDist>::my_heap;
};
struct min_heap : my_heap<CompMinDist> {
    using my_heap<CompMinDist>::my_heap;
};

#ifdef _OPENMP
thread_local std::mt19937 thread_rng;
#endif
}

struct HNSWIndex : Index {
    int dim;

    std::vector<Vertex> vertices;
    std::vector<VertexPtr> all_neighbors;
    int maxVertices;
    int maxDegree;
    double m_L;
    int efConstruction;

    std::vector<int> hop_list;
    int num_hops;

    VertexPtr entry = nullptr;
    int max_level = -1;
#ifdef _OPENMP
    std::shared_mutex entry_mutex;
    std::mutex visited_mutex;
    std::shared_mutex vertices_mutex;
    std::vector<std::unique_ptr<std::shared_mutex>> neighbor_mutexes;
    std::unique_ptr<std::atomic_flag[]> below_cvs;
#endif

    std::mt19937 rng;
    std::vector<VisitedMap> visited_pool;

    HNSWIndex(int dim, int maxVertices, int maxDegree, int efConstruction)
            : dim(dim), maxVertices(maxVertices), maxDegree(maxDegree), m_L(1.0 / std::log((double) maxDegree)), efConstruction(efConstruction) {
        rng.seed(0xdeadbeef);

        hop_list.reserve(10000);

#ifdef _OPENMP
        neighbor_mutexes.reserve(2 * maxVertices);

#pragma omp parallel
        {
            std::array<std::seed_seq::result_type, std::mt19937::state_size>
                seed_arr;
#pragma omp critical(access_rng)
            {
                for (auto &x : seed_arr) {
                    x = rng();
                }
            }
            std::seed_seq seed_seq(seed_arr.begin(), seed_arr.end());
            thread_rng.seed(seed_seq);
        }
#endif

        // extremely conservative bound
        vertices.reserve(2 * maxVertices);
#ifdef _OPENMP
        below_cvs = std::make_unique<std::atomic_flag[]>(maxVertices);
#endif
        for (int i = 0; i < maxVertices; i++) {
            vertices.emplace_back(i, nullptr);
#ifdef _OPENMP
            neighbor_mutexes.push_back(std::make_unique<std::shared_mutex>());
            below_cvs[i].clear();
#endif
        }
        // maxVertices * 2 * maxDegree + maxVertices * maxDegree
        all_neighbors.reserve(3 * (size_t) maxVertices * (size_t) maxDegree);
        all_neighbors.resize((size_t) maxVertices * 2 * (size_t) maxDegree);
        // initialize visited maps
        // if multi-threaded, we should initialize one for each thread
        int pool_size;
#ifdef _OPENMP
        pool_size = omp_get_max_threads();
#else
        pool_size = 1;
#endif
        for (int i = 0; i < pool_size; i++) {
            visited_pool.emplace_back(maxVertices);
        }
    }

    int getRandomLevel() {
#ifdef _OPENMP
        auto &rng = thread_rng;
#endif
        double rand = std::generate_canonical<double, 50>(rng);
        if (rand == 0.0) rand = std::numeric_limits<double>::epsilon();
        double l = -std::log(rand) * m_L;
        if (l > 40.0) return 40;
        return (int) l;
    }

    Vertex &get(VertexPtr p) { return vertices[p.i]; }
    Vertex &get(Vertex *v) const { return *v; }
    VertexPtr makeVertex(int id, float *data, int level) {
        if (level == 0) {
            WITH_SHARED_LOCK(vertices_mutex);
            vertices[id].data = data;
            assert(vertices[id].numNeighbors == 0);
            return VertexPtr(id);
        }
        WITH_UNIQUE_LOCK(vertices_mutex);
        VertexPtr p((uint32_t) vertices.size());
        vertices.emplace_back(id, data);
        size_t new_size = (size_t) maxVertices * (size_t) maxDegree
                             + vertices.size() * (size_t) maxDegree;
        assert(new_size == all_neighbors.size() + (size_t) maxDegree);
        all_neighbors.resize(new_size);
#ifdef _OPENMP
        neighbor_mutexes.push_back(std::make_unique<std::shared_mutex>());
#endif
        assert(vertices[id].numNeighbors == 0);
        return p;
    }
    bool isLayer0(VertexPtr p) {
        return (int) p.i < maxVertices;
    }
    std::span<VertexPtr> getNeighbors(VertexPtr p) {
        if (isLayer0(p)) {
            return std::span<VertexPtr>(&all_neighbors[2 * maxDegree * p.i],
                                        get(p).numNeighbors);
        }
        int idx = 2 * maxDegree * maxVertices
            + maxDegree * (p.i - maxVertices);
        return std::span<VertexPtr>(&all_neighbors[idx],
                                    get(p).numNeighbors);
    }
    void pushNeighbor(VertexPtr p, VertexPtr n) {
        if (isLayer0(p)) {
            assert(get(p).numNeighbors < 2 * maxDegree);
            all_neighbors[2 * maxDegree * p.i + get(p).numNeighbors++] = n;
            return;
        }
        assert(get(p).numNeighbors < maxDegree);
        int idx = 2 * maxDegree * maxVertices
            + maxDegree * (p.i - maxVertices);
        all_neighbors[idx + get(p).numNeighbors++] = n;
    }

    template<bool Threaded=DEFAULT_THREADED>
    VisitedMap takeVisitedMap() {
        std::conditional_t<Threaded, UNIQUE_LOCK_TY(std::mutex), int> my_lock;
        if constexpr (Threaded) {
            my_lock = UNIQUE_LOCK(std::mutex)(visited_mutex);
        } else { (void) my_lock; }
        return VisitedMap::takeFromPool(visited_pool, maxVertices);
    }

    template<bool Threaded=DEFAULT_THREADED>
    void releaseVisitedMap(VisitedMap &&map) {
        std::conditional_t<Threaded, UNIQUE_LOCK_TY(std::mutex), int> my_lock;
        if constexpr (Threaded) {
            my_lock = UNIQUE_LOCK(std::mutex)(visited_mutex);
        } else { (void) my_lock; }
        visited_pool.push_back(std::move(map));
    }

    float computeDistance(float *a, float *b) { return simd_l2dist(a, b, dim); }

    template<typename Cont, typename Comp>
    void pushq(std::priority_queue<PQElement, Cont, Comp> &pq,
               VertexPtr ptr, float *q) {
        pushq(pq, ptr, computeDistance(get(ptr).data, q));
    }

    template<typename Cont, typename Comp>
    void pushq(std::priority_queue<PQElement, Cont, Comp> &pq,
               VertexPtr ptr, float dist) {
        pq.emplace(dist, ptr);
    }

    void insert(int id, float *data);
    template<bool Threaded=DEFAULT_THREADED>
    max_heap searchLayer(float *data, max_heap ep, int ef);
    template<bool Threaded=DEFAULT_THREADED>
    PQElement searchLayer1(float *data, PQElement ep);
    std::vector<PQElement> selectNeighbors(std::vector<PQElement> candidates, int M);
    std::vector<PQElement> query(float *data, int k, int ef);
};

template<bool Threaded>
max_heap HNSWIndex::searchLayer(float *data, max_heap ep, int ef) {
    if (ep.empty()) return ep;

    VisitedMap visited = takeVisitedMap();

    min_heap candidates(ep);
    max_heap nearest = std::move(ep);
    float farthest_dist = nearest.top().dist;

    std::conditional_t<Threaded, std::vector<VertexPtr>, std::span<VertexPtr>>
        the_neighbors;

    if constexpr (Threaded) {
        the_neighbors.reserve(maxDegree);
    }

    while (!candidates.empty()) {
        PQElement cur = candidates.top();
        candidates.pop();
        if (cur.dist > farthest_dist) {
            break;
        }
        num_hops++;
        if constexpr (Threaded) {
            SHARED_LOCK_TY(std::shared_mutex) COND_LOCK(*neighbor_mutexes[cur.vertex.i]);
            auto span = getNeighbors(cur.vertex);
            the_neighbors.assign(span.begin(), span.end());
        } else {
            the_neighbors = getNeighbors(cur.vertex);
        }
        for (VertexPtr e : the_neighbors) {
            assert(e);
            int id = get(e).id;
            if (visited[id]) continue;
            visited.set(id);
            float dist = computeDistance(get(e).data, data);
            if (dist < farthest_dist || candidates.size() < (size_t) ef) {
                pushq(candidates, e, dist);
                if (nearest.size() >= (size_t) ef) {
                    if (dist < farthest_dist) {
                        nearest.pop();
                        pushq(nearest, e, dist);
                        farthest_dist = nearest.top().dist;
                    }
                } else {
                    pushq(nearest, e, dist);
                    farthest_dist = nearest.top().dist;
                }
            }
        }
    }

    return nearest;
}

template<bool Threaded>
PQElement HNSWIndex::searchLayer1(float *data, PQElement ep) {
    VisitedMap visited = takeVisitedMap();

    min_heap candidates; candidates.push(ep);
    PQElement nearest = ep;

    std::conditional_t<Threaded, std::vector<VertexPtr>, std::span<VertexPtr>>
        the_neighbors;

    while (!candidates.empty()) {
        PQElement cur = candidates.top();
        candidates.pop();
        if (cur.dist > nearest.dist) {
            break;
        }
        num_hops++;
        if constexpr (Threaded) {
            SHARED_LOCK_TY(std::shared_mutex) COND_LOCK(*neighbor_mutexes[cur.vertex.i]);
            auto span = getNeighbors(cur.vertex);
            the_neighbors.assign(span.begin(), span.end());
        } else {
            the_neighbors = getNeighbors(cur.vertex);
        }
        for (VertexPtr e : the_neighbors) {
            int id = get(e).id;
            if (visited[id]) continue;
            visited.set(id);
            float dist = computeDistance(get(e).data, data);
            if (dist < nearest.dist || candidates.empty()) {
                pushq(candidates, e, dist);
                if (dist < nearest.dist) {
                    nearest = PQElement(dist, e);
                }
            }
        }
    }

    return nearest;
}


std::string out_fn_hnsw(Index *raw_index, argparse::ArgumentParser *) {
    HNSWIndex *index = (HNSWIndex *) raw_index;
    std::stringstream out;
    out << "out_hnsw_M" << index->maxDegree << "_efC" << index->efConstruction;
    return out.str();
}

void make_arg_parser_hnsw(argparse::ArgumentParser &parser) {
    parser.add_argument("-M", "--max-degree")
        .help("maximum degree for graph vertices (M_max); doubled for layer 0")
        .default_value(32)
        .scan<'d', int>();
    parser.add_argument("--ef-construction")
        .help("exploration factor used during index construction")
        .default_value(128)
        .scan<'d', int>();
}

std::unique_ptr<Index> preprocess_ann_hnsw(bool is_l2, RawVectorData *vectors, RawVectorData *learn, argparse::ArgumentParser &parser) {
    if (!is_l2) {
        throw std::runtime_error("we only support L2 for HNSW");
    }
    (void) learn;
    int maxDegree =
        parser.get<int>("--max-degree");
    int efConstruction =
        parser.get<int>("--ef-construction");

    ScopedTimer timer("index construction");
    auto ret = std::make_unique<HNSWIndex>(
        vectors->dim, vectors->length,
        maxDegree,
        efConstruction
    );
    int i_start = 0;
#ifdef _OPENMP
    // insert some vertices sequentially first
    for (i_start = 0; i_start < 1000 && i_start < vectors->length; i_start++) {
        float *data = &vectors->at(i_start, 0);
        ret->insert(i_start, data);
    }
    std::atomic<int>
#else
    int
#endif
        insert_progress = i_start;
#pragma omp parallel for
    for (int i = i_start; i < vectors->length; i++) {
        int cur_progress =
#ifdef _OPENMP
            insert_progress.fetch_add(1);
#else
            insert_progerss++;
#endif
        if (cur_progress % 1000 == 0) {
#pragma omp critical(cout)
            std::cout << "insert progress: " << cur_progress << "/" << vectors->length << std::endl;
        }
        float *data = &vectors->at(i, 0);
        ret->insert(i, data);
    }
    return ret;
}

int compute_ann_hnsw(RawVectorData *vectors, int k, float *query, int *result, Index *raw_index) {
    (void) vectors;
    HNSWIndex *index = (HNSWIndex *) raw_index;
    index->num_hops = 0;
    auto vec = index->query(query, k, k);
    for (int i = 0; i < (int) vec.size(); i++) {
        result[i] = index->get(vec[i].vertex).id;
    }
    index->hop_list.push_back(index->num_hops);
    return vec.size();
}

void output_stats_hnsw(RawVectorData *vectors, int k, RawVectorData *queries, int *result, Index *index) {
    (void) vectors; (void) k; (void) result;
    size_t total_hops = 0;
    auto hop_list = static_cast<HNSWIndex *>(index)->hop_list;
    for (int h : hop_list) total_hops += h;
    std::cout << "[STATS] avg num vertices explored: " << std::setprecision(6) << (double) total_hops / queries->length << std::endl;

    std::stringstream out_fn;
    out_fn << "hopdist_" << out_fn_hnsw(index, nullptr);
    std::ofstream f(out_fn.str(), std::ios::binary);
    f.write((const char *) hop_list.data(), hop_list.size() * sizeof hop_list[0]);
}



void HNSWIndex::insert(int id, float *data) {
    max_heap ep;
    VertexPtr above = nullptr;
    VertexPtr root = nullptr;

    VertexPtr my_entry;
    int my_max_level;

    {
        WITH_SHARED_LOCK(entry_mutex);
        my_entry = entry;
        my_max_level = max_level;
    }

    int cur_level = my_max_level;
    int ins_level = getRandomLevel();
    if (ins_level > my_max_level) {
        for (cur_level = ins_level; cur_level > my_max_level; cur_level--) {
            VertexPtr q_ptr = makeVertex(id, data, cur_level);
            WITH_SHARED_LOCK(vertices_mutex);
            assert((!root) == (!above));
            if (!root) root = q_ptr;
            if (above) {
                get(above).below = q_ptr;
#ifdef _OPENMP
// #pragma omp critical(cout)
//                 std::cout << "set " << get(above).id << std::endl;
                below_cvs[get(above).id].test_and_set();
                below_cvs[get(above).id].notify_all();
#endif
            }
            above = q_ptr;
        }
    } else {
        assert(my_entry);
    }
    if (my_entry) {
        float dist;
        {
            WITH_SHARED_LOCK(vertices_mutex);
            dist = computeDistance(get(my_entry).data, data);
        }
        PQElement ep1(dist, my_entry);
        for (; cur_level > ins_level; cur_level--) {
            ep1 = searchLayer1(data, std::move(ep1));
            VertexPtr next;
            { WITH_SHARED_LOCK(vertices_mutex); next = get(ep1.vertex).below; }
#ifdef _OPENMP
            while (!next && cur_level > 0) {
// #pragma omp critical(cout)
//                 std::cout << "wait " << get(ep1.vertex).id << std::endl;
                below_cvs[get(ep1.vertex).id].wait(false);
                { WITH_SHARED_LOCK(vertices_mutex); next = get(ep1.vertex).below; }
            }
#endif
            ep1.vertex = next;
        }
        ep.push(std::move(ep1));
    } else {
        cur_level = std::min(ins_level, cur_level);
    }
    for (; cur_level >= 0; cur_level--) {
        VertexPtr q_ptr = makeVertex(id, data, cur_level);
        WITH_SHARED_LOCK(vertices_mutex);
        Vertex &q = get(q_ptr);
        assert((size_t) q.numNeighbors == 0);
        ep = searchLayer(data, std::move(ep), efConstruction);
        assert((size_t) q.numNeighbors == 0);
        int M = cur_level == 0 ? 2 * maxDegree : maxDegree;
        assert((size_t) q.numNeighbors == 0);
        auto neighbors = selectNeighbors(ep.elements(), M);
        assert((size_t) q.numNeighbors == 0);
        (void) q;
        for (auto &n : neighbors) {
            // safety: q is not yet in the graph
            pushNeighbor(q_ptr, n.vertex);
        }
        assert((size_t) q.numNeighbors == neighbors.size());
        if (above) {
            get(above).below = q_ptr;
#ifdef _OPENMP
// #pragma omp critical(cout)
//             std::cout << "set " << get(above).id << std::endl;
            below_cvs[get(above).id].test_and_set();
            below_cvs[get(above).id].notify_all();
#endif
        }
        for (auto &n : neighbors) {
            // we hold only one lock, so no deadlocks between neighbor_mutexes
            // lock ordering: vertices_mutex -> (any) neighbor_mutexes
            WITH_UNIQUE_LOCK(*neighbor_mutexes[n.vertex.i]);
            Vertex &e = get(n.vertex);
            if (e.numNeighbors < M) {
                pushNeighbor(n.vertex, q_ptr);
                continue;
            }
            std::vector<PQElement> temp;
            for (VertexPtr v : getNeighbors(n.vertex)) {
                temp.emplace_back(computeDistance(get(v).data, e.data), v);
            }
            temp.emplace_back(n.dist, q_ptr);
            auto new_neighbors = selectNeighbors(std::move(temp), M);
            e.numNeighbors = new_neighbors.size();
            auto en = getNeighbors(n.vertex);
            for (int i = 0; i < (int) new_neighbors.size(); i++) {
                en[i] = new_neighbors[i].vertex;
                assert(en[i]);
            }
        }
        my_lock.unlock();
        above = q_ptr;
        for (auto &el : ep.elements_for_lower()) {
            VertexPtr next;
            { WITH_SHARED_LOCK(vertices_mutex); next = get(el.vertex).below; }
#ifdef _OPENMP
            while (!next && cur_level > 0) {
// #pragma omp critical(cout)
//                 std::cout << "wait " << get(el.vertex).id << std::endl;
                below_cvs[get(el.vertex).id].wait(false);
                { WITH_SHARED_LOCK(vertices_mutex); next = get(el.vertex).below; }
            }
#endif
            el.vertex = next;
        }
    }
    if (ins_level > my_max_level) {
        WITH_UNIQUE_LOCK(entry_mutex);
        if (ins_level > max_level) {
            max_level = ins_level;
            if (!root) [[unlikely]] throw std::runtime_error("root should exist");
            entry = root;
        }
    }
}

std::vector<PQElement> HNSWIndex::selectNeighbors(std::vector<PQElement> candidates, int M) {
    if (candidates.size() < (size_t) M) return candidates;
    std::sort(candidates.begin(), candidates.end(), CompareDistance());
    int o = 0;
    for (int i = 0; i < (int) candidates.size(); i++) {
        bool ok = true;
        float q_dist = candidates[i].dist;
        float *cand_data = get(candidates[i].vertex).data;
        for (int j = 0; j < o; j++) {
            float j_dist = computeDistance(cand_data, get(candidates[o].vertex).data);
            if (j_dist < q_dist) {
                ok = false;
                break;
            }
        }
        if (ok) {
            candidates[o] = std::move(candidates[i]);
            o++;
            if (o >= M) break;
        }
    }
    candidates.erase(candidates.begin() + o, candidates.end());
    return candidates;
}

std::vector<PQElement> HNSWIndex::query(float *data, int k, int ef) {
    PQElement ep1(computeDistance(get(entry).data, data), entry);
    int cur_level = max_level;
    for (; cur_level > 0; cur_level--) {
        ep1 = searchLayer1<false>(data, std::move(ep1));
        ep1.vertex = get(ep1.vertex).below;
    }
    max_heap ep;
    ep.push(std::move(ep1));
    max_heap nearest = searchLayer<false>(data, std::move(ep), ef);
    while (nearest.size() > (size_t) k) {
        nearest.pop();
    }
    std::vector<PQElement> ret = std::move(nearest).move_elements();
    static_assert(std::is_same<decltype(nearest)::value_compare, CompareDistance>::value, "must sort using same comparator");
    std::sort_heap(ret.begin(), ret.end(), CompareDistance());
    return ret;
}
