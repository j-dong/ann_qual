#include "kernel_vamana.h"
#include "kernel_utils.h"
#include "load_files.h"
#include "timer.h"
#include "simd_utils.h"
#include "graph_utils.h"
#include "visited_map.h"
#include <span>

#ifndef NOMINMAX
# define NOMINMAX 1
#endif
#include "inc_cblas.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "argparse/argparse.hpp"

#ifdef _OPENMP
# include <atomic>
# include <mutex>
# include <shared_mutex>
# define WITH_SHARED_LOCK(mutex) std::shared_lock my_lock(mutex)
# define WITH_SHARED_LOCK_2(mutex) std::shared_lock my_lock_2(mutex)
# define WITH_UNIQUE_LOCK(mutex) std::unique_lock my_lock(mutex)
# define WITH_UNIQUE_LOCK_2(mutex) std::unique_lock my_lock_2(mutex)
# define DEFAULT_THREADED true
# define UNIQUE_LOCK(Ty) std::unique_lock<Ty>
# define UNIQUE_LOCK_TY(Ty) std::unique_lock<Ty>
# define SHARED_LOCK_TY(Ty) std::unique_lock<Ty>
#else
# define DEFAULT_THREADED false
# define UNIQUE_LOCK_TY(Ty) void
# define UNIQUE_LOCK(Ty) UNIQUE_LOCK_M
# define SHARED_LOCK_TY(Ty) UNIQUE_LOCK_M
# define UNIQUE_LOCK_M(val) 0
namespace {
    template<typename T>
    struct UNIQUE_LOCK_M {};
}
#endif

namespace {
struct Vertex {
    int numNeighbors = 0;
    float *data;

    Vertex(float *data) : data(data) {}
};

struct max_heap : vertex_heap<CompMaxDist> {
    using vertex_heap<CompMaxDist>::vertex_heap;
};
struct min_heap : vertex_heap<CompMinDist> {
    using vertex_heap<CompMinDist>::vertex_heap;
};

struct CompPairVPFirst {
    using A = VertexPtr;
    template<typename B>
    bool operator()(const std::pair<A, B> &a, const std::pair<A, B> &b) const {
        return a.first.i < b.first.i;
    }
};

struct ManualDeleter {
    template<typename T>
    void operator()(T p[]) const { operator delete[](p); }
};

class bounded_pq;
struct pq_inserter {
    int start;
    bounded_pq &heap;

    pq_inserter(int start, bounded_pq &heap)
        : start(start), heap(heap) {}

    void emplace(float dist, VertexPtr ptr);
    void emplace_unchecked(float dist, VertexPtr ptr);
    ~pq_inserter() noexcept(false);
};

// max-heap, but iterates to find/remove the min element
class bounded_pq {
protected:
    int capacity;
    int i_cur; int i_end;
    int c_cap, temp_cap;
    PQElement *c, *temp;

public:
    bounded_pq(int capacity) : capacity(capacity), i_cur(0), i_end(0) {
        if (capacity == 0) throw std::invalid_argument("capacity cannot be 0");
        c_cap = capacity; temp_cap = capacity;
        c = (PQElement *) malloc(sizeof(PQElement) * (size_t) capacity);
        temp = (PQElement *) malloc(sizeof(PQElement) * (size_t) capacity);
    }

    bounded_pq(bounded_pq &&o)
        : capacity(o.capacity), i_cur(o.i_cur), i_end(o.i_end),
          c_cap(o.c_cap), temp_cap(o.temp_cap),
          c(o.c), temp(o.temp) {
        o.c = nullptr; o.temp = nullptr;
        o.c_cap = 0; o.temp_cap = 0; o.i_cur = o.i_end = 0;
    }

    bounded_pq &operator=(bounded_pq &&o) {
        if (&o == this) return *this;
        free(c); free(temp);
        std::memcpy(this, &o, sizeof *this);
        o.c = nullptr; o.temp = nullptr;
        o.c_cap = 0; o.temp_cap = 0; o.i_cur = o.i_end = 0;
        return *this;
    }

    ~bounded_pq() {
        free(c); free(temp);
    }

    void emplace_empty(float dist, VertexPtr ptr) {
        if (!empty()) throw std::domain_error("emplace_empty but not empty");
        c[i_end++] = PQElement(dist, ptr);
    }

    bool empty() { return i_end == i_cur; }
    int size() { return i_end - i_cur; }

    const PQElement *begin() { return &c[0]; }
    const PQElement *end() { return &c[i_end]; }
    const PQElement *end_or(int k) { if (k > i_end) k = i_end; return &c[k]; }

    void reserve_more(size_t extra) {
        int new_cap = i_end + extra;
        if (new_cap <= c_cap) return;
        new_cap = std::max(new_cap, 2 * c_cap);
        PQElement *buf = (PQElement *) malloc(sizeof(PQElement) * (size_t) new_cap);
        if (!buf) {
            throw std::bad_alloc();
        }
        memcpy(buf, c, i_end * sizeof(PQElement));
        free(c);
        c = buf;
    }

    pq_inserter do_insert() { return pq_inserter(i_end, *this); }

    PQElement pop_min() {
        if (empty()) throw std::domain_error("pop from empty pq");
        return c[i_cur++];
    }

protected:
    friend struct pq_inserter;
};

inline void pq_inserter::emplace(float dist, VertexPtr ptr) {
    heap.reserve_more(1);
    emplace_unchecked(dist, ptr);
}

inline void pq_inserter::emplace_unchecked(float dist, VertexPtr ptr) {
    heap.c[heap.i_end++] = PQElement(dist, ptr);
}

inline pq_inserter::~pq_inserter() noexcept(false) {
    assert(heap.i_end >= start);
    // precondition: heap.c[0:start] is sorted by distance
    // precondition: new elements are stored starting from start
    // precondition: elements up to heap.i_cur are visited
    // postcondition: heap.c is sorted by distance
    // postcondition: heap.c contains the at most heap.capacity
    //                closest elements from initial heap.c
    // postcondition: elements up to heap.i_cur are visited
    //                (some visited elements may be re-inserted)
    PQElement *it_start = heap.c;
    PQElement *it_cur = heap.c + heap.i_cur;
    PQElement *it_mid = heap.c + start;
    PQElement *__restrict it_end = heap.c + heap.i_end;
    std::sort(it_mid, it_end, CompareDistance());
    int new_size = std::min(heap.capacity, heap.size());
    if (new_size < 0) __builtin_unreachable();
    if (new_size >= heap.temp_cap) { [[unlikely]]
        free(heap.temp); heap.temp = nullptr;
        int new_cap = std::max(new_size, heap.temp_cap * 2);
        heap.temp = (PQElement *) malloc(sizeof(PQElement) * new_cap);
        if (!heap.temp) throw std::bad_alloc();
        heap.temp_cap = new_cap;
    }
    {
        PQElement *__restrict it_1 = it_start;
        PQElement *__restrict it_2 = it_mid;
        int size = 0;
        while (it_1 < it_cur && it_2 < it_end && size < heap.capacity && it_1->dist <= it_2->dist) {
            heap.temp[size++] = *it_1++;
        }
        heap.i_cur = size;
        while (it_1 < it_mid && it_2 < it_end && size < heap.capacity) {
            if (it_1->dist <= it_2->dist) {
                heap.temp[size++] = *it_1++;
            } else {
                heap.temp[size++] = *it_2++;
            }
        }
        if (it_1 == it_mid && size < heap.capacity) {
            int num = std::min((int) (it_end - it_2), heap.capacity - size);
            std::memcpy(&heap.temp[size], it_2,
                        sizeof(PQElement) * num);
            size += num;
        }
        if (it_2 == it_end && size < heap.capacity) { // (technically disjoint)
            int num = std::min((int) (it_mid - it_1), heap.capacity - size);
            std::memcpy(&heap.temp[size], it_1,
                        sizeof(PQElement) * num);
            size += num;
        }
        heap.i_end = size;
    }
    std::swap(heap.temp, heap.c);
    std::swap(heap.temp_cap, heap.c_cap);
}
}

struct VamanaIndex : Index {
    int dim;

    std::vector<Vertex> vertices;
    std::vector<VertexPtr> all_neighbors;
    std::vector<VertexPtr> visit_order;
    int maxVertices;
    int maxDegree;
    int L;
    RawVectorData *raw_data;

    std::vector<int> hop_list;
    int num_hops;

    VertexPtr entry = nullptr;

    std::mt19937 rng;
    std::vector<VisitedMap> visited_pool;

#ifdef _OPENMP
    std::mutex visited_mutex;
#endif

    VamanaIndex(int maxDegree, int L,
                RawVectorData *data)
            : dim(data->dim), maxVertices(data->length), maxDegree(maxDegree), L(L), raw_data(data) {
        rng.seed(0xdeadbeef);
        vertices.reserve(maxVertices);
        all_neighbors.resize(maxVertices * maxDegree);
        for (int i = 0; i < maxVertices; i++) {
            vertices.emplace_back(&data->at(i, 0));
        }
        for (int i = 0; i < 1; i++) {
            visited_pool.emplace_back(maxVertices);
        }
        hop_list.reserve(10000);
    }

    Vertex &get(VertexPtr p) { return vertices[p.i]; }

    std::span<VertexPtr> neighbors(VertexPtr p) {
        return std::span<VertexPtr>(&all_neighbors[maxDegree * p.i],
                                    get(p).numNeighbors);
    }

    void push_neighbor(VertexPtr p, VertexPtr n) {
        assert(get(p).numNeighbors < maxDegree);
        all_neighbors[maxDegree * p.i + get(p).numNeighbors++] = n;
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

    template<typename Queue>
    void pushq(Queue &pq, VertexPtr ptr, float *q) {
        pushq(pq, ptr, computeDistance(get(ptr).data, q));
    }

    template<typename Queue>
    void pushq(Queue &pq, VertexPtr ptr, float dist) {
        pq.emplace(dist, ptr);
    }

    void initialize();
    std::vector<PQElement> greedySearch(float *data, int k);
    PQElement greedySearch(VertexPtr p, std::vector<PQElement> &out_visited
#ifdef _OPENMP
                           , std::shared_mutex *mutexes
#endif
    );
    int distance(VertexPtr v);
    void robustPrune(VertexPtr p, std::vector<PQElement> &out, float alpha);
    void refine(float alpha);
};

void VamanaIndex::initialize() {
    // set edges to random neighbors
    VisitedMap visited = takeVisitedMap();
    if (maxDegree >= maxVertices) {
        throw std::runtime_error("cannot have degree >= max vertices");
    }
    std::uniform_int_distribution<> rand_vert(0, maxVertices - 1);
    for (int i = 0; i < maxVertices; i++) {
        VertexPtr p = VertexPtr(i);
        Vertex &v = get(p);
        v.numNeighbors = maxDegree;
        auto n = neighbors(p);
        visited.set(i);
        for (int e = 0; e < maxDegree; e++) {
            int j;
            do {
                j = rand_vert(rng);
            } while (visited[j]);
            visited.set(j);
            n[e] = VertexPtr(j);
        }
        visited.reset();
    }
    // compute medioid
    // using procedure in reference code
    std::vector<float> center;
    center.resize(dim);
    std::vector<float> ref;
    ref.resize(maxVertices);
    std::fill(ref.begin(), ref.end(), 1.0f / maxVertices);
    cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        maxVertices,
        dim,
        1.0,
        &raw_data->at(0, 0),
        dim + 1,
        ref.data(),
        1,
        0.0f,
        center.data(),
        1
    );
    // now reuse ref as distances
    for (int i = 0; i < maxVertices; i++) {
        ref[i] = -computeDistance(&raw_data->at(i, 0), center.data());
    }
    entry = VertexPtr(simd_argmax(ref.data(), maxVertices));
}

std::vector<PQElement> VamanaIndex::greedySearch(float *data, int k) {
    bounded_pq search_list(L);
    search_list.emplace_empty(computeDistance(get(entry).data, data), entry);
    VisitedMap visited = takeVisitedMap<false>();
    // use tag + 1 to store inserted
    while (!search_list.empty()) {
        PQElement cur = search_list.pop_min();
        if (visited[cur.vertex.i]) {
            continue;
        }
        visited.set(cur.vertex.i);
        num_hops++;
        search_list.reserve_more(neighbors(cur.vertex).size());
        auto ins = search_list.do_insert();
        for (auto n : neighbors(cur.vertex)) {
            if (visited[n.i] || visited.vec[n.i] == visited.tag + 1) continue;
            ins.emplace_unchecked(computeDistance(get(n).data, data), n);
            visited.vec[n.i] = visited.tag + 1;
        }
    }
    visited.reset(); // for extra +1
    releaseVisitedMap<false>(std::move(visited));
    return std::vector(search_list.begin(), search_list.end_or(k));
}

PQElement VamanaIndex::greedySearch(VertexPtr p, std::vector<PQElement> &out_visited
#ifdef _OPENMP
                           , std::shared_mutex *mutexes
#endif
) {
    float *data = get(p).data;
    bounded_pq search_list(L);
    search_list.emplace_empty(computeDistance(get(entry).data, data), entry);
    VisitedMap visited = takeVisitedMap();

#ifdef _OPENMP
        std::vector<VertexPtr> the_neighbors;
        the_neighbors.reserve(maxDegree);
#else
        std::span<VertexPtr> the_neighbors;
#endif

    // +0 -> visited
    // +1 -> inserted, p's neighbor
    // +2 -> inserted
    bool found_p = false;
    while (!search_list.empty()) {
        PQElement cur = search_list.pop_min();
        if (visited[cur.vertex.i]) {
            continue;
        }
        if (visited.vec[cur.vertex.i] != visited.tag + 1) {
            out_visited.push_back(cur);
        }
        visited.set(cur.vertex.i);

#if _OPENMP
        {
            std::shared_lock cur_lock((mutexes[cur.vertex.i]));
            auto span = neighbors(cur.vertex);
            the_neighbors.assign(span.begin(), span.end());
        }
#else
        the_neighbors = neighbors(cur.vertex);
#endif

        search_list.reserve_more(the_neighbors.size());
        auto ins = search_list.do_insert();
        if (p == cur.vertex) {
            found_p = true;
            for (auto n : the_neighbors) {
                auto nt = visited.vec[n.i];
                if (nt == visited.tag || nt == visited.tag + 1) continue;
                visited.vec[n.i] = visited.tag + 1;
                float dist = computeDistance(get(n).data, data);
                out_visited.emplace_back(dist, n);
                if (nt == visited.tag + 2) continue;
                ins.emplace_unchecked(dist, n);
            }
        } else {
            for (auto n : the_neighbors) {
                if (visited[n.i] || visited.vec[n.i] == visited.tag + 1 || visited.vec[n.i] == visited.tag + 2) continue;
                ins.emplace_unchecked(computeDistance(get(n).data, data), n);
                visited.vec[n.i] = visited.tag + 2;
            }
        }
    }
    if (!found_p) {
#ifdef _OPENMP
        std::shared_lock p_lock(mutexes[p.i]);
#endif
        for (auto n : neighbors(p)) {
            if (visited[n.i]) continue;
            float dist = computeDistance(get(n).data, data);
            out_visited.emplace_back(dist, n);
        }
    }
    visited.reset();
    visited.reset();
    releaseVisitedMap(std::move(visited));
    return *search_list.begin();
}

void VamanaIndex::robustPrune(VertexPtr p, std::vector<PQElement> &out, float alpha) {
    VisitedMap to_remove = takeVisitedMap();
    to_remove.set(p.i);
    std::sort(out.begin(), out.end(), CompareDistance());
    int count = 0;
    auto end = out.end();
    auto it = out.begin();
    for (; it != end; ++it) {
        auto &cur = *it;
        if (to_remove[cur.vertex.i]) continue;
        count++;
        if (count >= maxDegree) break;
        for (auto ot = it + 1; ot != end; ++ot) {
            auto &other = *ot;
            float dist = computeDistance(get(cur.vertex).data, get(other.vertex).data);
            if (alpha * dist <= other.dist) {
                to_remove.set(other.vertex.i);
            }
        }
    }
    out.erase(
        std::remove_if(out.begin(), it,
                       [&to_remove](PQElement &e) { return to_remove[e.vertex.i]; }),
        out.end());
    releaseVisitedMap(std::move(to_remove));
}

#ifdef _OPENMP

void VamanaIndex::refine(float alpha) {
    {
        ScopedTimer timer("shuffle visit order");
        visit_order.resize(maxVertices);
        for (int i = 0; i < maxVertices; i++) {
            visit_order[i].i = i;
        }
        std::shuffle(visit_order.begin(), visit_order.end(), rng);
    }
    std::unique_ptr<std::shared_mutex[]> mutexes = std::make_unique<std::shared_mutex[]>(
        maxVertices
    );
    std::atomic<int> total_progress = 0;
#pragma omp parallel
    {
    std::vector<PQElement> p_visited;
    std::vector<PQElement> temp_neighbors;
    p_visited.reserve(L * 2);
    temp_neighbors.reserve(maxDegree + 1);
    int my_progress;
#pragma omp for
    for (size_t vi = 0; vi < visit_order.size(); vi++) {
        if ((my_progress = total_progress.fetch_add(1)) % 1000 == 0) {
#pragma omp critical
            std::cout << "refine (alpha=" << alpha << ") progress: " << my_progress << "/" << visit_order.size() << std::endl;
        }
        VertexPtr p = visit_order[vi];
        p_visited.clear();
        greedySearch(p, p_visited
#ifdef _OPENMP
                     , mutexes.get()
#endif
        );
        robustPrune(p, p_visited, alpha);
        {
            std::unique_lock p_lock(mutexes[p.i]);
            get(p).numNeighbors = p_visited.size();
            auto pn = neighbors(p);
            for (int i = 0; i < (int) p_visited.size(); i++) {
                pn[i] = p_visited[i].vertex;
            }
        }
        for (PQElement &je : p_visited) {
            VertexPtr j = je.vertex;
            std::unique_lock j_lock(mutexes[j.i]);
            auto jn = neighbors(j);
            if ((int) jn.size() < maxDegree) {
                push_neighbor(j, p);
            } else {
                temp_neighbors.clear();
                for (VertexPtr n : jn) {
                    temp_neighbors.emplace_back(computeDistance(get(n).data, get(j).data), n);
                }
                temp_neighbors.emplace_back(je.dist, p);
                robustPrune(j, temp_neighbors, alpha);
                assert((int) temp_neighbors.size() <= maxDegree);
                get(j).numNeighbors = temp_neighbors.size();
                jn = neighbors(j);
                for (int i = 0; i < (int) temp_neighbors.size(); i++) {
                    jn[i] = temp_neighbors[i].vertex;
                }
            }
        }
    }
    }
}

#else

void VamanaIndex::refine(float alpha) {
    if (visit_order.empty()) {
        ScopedTimer timer("shuffle visit order");
        visit_order.resize(maxVertices);
        for (int i = 0; i < maxVertices; i++) {
            visit_order[i].i = i;
        }
        std::shuffle(visit_order.begin(), visit_order.end(), rng);
    }
    std::vector<PQElement> p_visited;
    std::vector<PQElement> temp_neighbors;
    p_visited.reserve(L * 2);
    temp_neighbors.reserve(maxDegree + 1);
    for (size_t vi = 0; vi < visit_order.size(); vi++) {
        if (vi % 1000 == 0) {
            std::cout << "refine (alpha=" << alpha << ") progress: " << vi << "/" << visit_order.size() << std::endl;
        }
        VertexPtr p = visit_order[vi];
        p_visited.clear();
        greedySearch(p, p_visited);
        robustPrune(p, p_visited, alpha);
        get(p).numNeighbors = (int) p_visited.size();
        auto pn = neighbors(p);
        for (int i = 0; i < (int) p_visited.size(); i++) {
            pn[i] = p_visited[i].vertex;
        }
        for (PQElement &je : p_visited) {
            VertexPtr j = je.vertex;
            auto jn = neighbors(j);
            if ((int) jn.size() < maxDegree) {
                push_neighbor(j, p);
            } else {
                temp_neighbors.clear();
                for (VertexPtr n : jn) {
                    temp_neighbors.emplace_back(computeDistance(get(n).data, get(j).data), n);
                }
                temp_neighbors.emplace_back(je.dist, p);
                robustPrune(j, temp_neighbors, alpha);
                assert((int) temp_neighbors.size() <= maxDegree);
                get(j).numNeighbors = temp_neighbors.size();
                jn = neighbors(j);
                for (int i = 0; i < (int) temp_neighbors.size(); i++) {
                    jn[i] = temp_neighbors[i].vertex;
                }
            }
        }
    }
}

#endif


std::string out_fn_vamana(Index *raw_index, argparse::ArgumentParser *) {
    VamanaIndex *index = (VamanaIndex *) raw_index;
    std::stringstream out;
    out << "out_vamana_R" << index->maxDegree << "_L" << index->L;
    return out.str();
}

void make_arg_parser_vamana(argparse::ArgumentParser &parser) {
    parser.add_argument("-R", "--max-degree")
        .help("maximum degree of graph vertices (R)")
        .default_value(128)
        .scan<'d', int>();
    parser.add_argument("-L", "--search-list-size")
        .help("search list size (L)")
        .default_value(256)
        .scan<'d', int>();
}

std::unique_ptr<Index> preprocess_ann_vamana(bool is_l2, RawVectorData *vectors, RawVectorData *learn, argparse::ArgumentParser &parser) {
    if (!is_l2) throw std::runtime_error("we only support L2 for Vamana");

    (void) learn;

    int maxDegree =
        parser.get<int>("--max-degree");
    int L =
        parser.get<int>("-L");

    ScopedTimer timer("index construction");
    auto ret = std::make_unique<VamanaIndex>(
        maxDegree,
        L,
        vectors);
    {
        ScopedTimer timer("initialize to random");
        ret->initialize();
    }
    {
        ScopedTimer timer("alpha = 1.0");
        ret->refine(1.0f);
    }
    {
        ScopedTimer timer("alpha = 2.0");
        ret->refine(2.0f);
    }
    return ret;
}

int compute_ann_vamana(RawVectorData *vectors, int k, float *query, int *result, Index *raw_index) {
    (void) vectors;
    VamanaIndex *index = (VamanaIndex *) raw_index;
    index->num_hops = 0;
    auto vec = index->greedySearch(query, k);
    for (int i = 0; i < (int) vec.size(); i++) {
        result[i] = vec[i].vertex.i;
    }
    index->hop_list.push_back(index->num_hops);
    return vec.size();
}

struct OutputData {
    int id;
    int level;
    int degree;
    int distance;
};

int VamanaIndex::distance(VertexPtr v) {
    float *data = get(v).data;
    bounded_pq search_list(L);
    search_list.emplace_empty(computeDistance(get(entry).data, data), entry);
    VisitedMap visited = takeVisitedMap<false>();
    int distance = 0;
    // use tag + 1 to store inserted
    while (!search_list.empty()) {
        PQElement cur = search_list.pop_min();
        if (cur.vertex == v) {
            return distance;
        }
        if (visited[cur.vertex.i]) {
            continue;
        }
        visited.set(cur.vertex.i);
        distance++;
        search_list.reserve_more(neighbors(cur.vertex).size());
        auto ins = search_list.do_insert();
        for (auto n : neighbors(cur.vertex)) {
            if (visited[n.i] || visited.vec[n.i] == visited.tag + 1) continue;
            ins.emplace_unchecked(computeDistance(get(n).data, data), n);
            visited.vec[n.i] = visited.tag + 1;
        }
    }
    visited.reset(); // for extra +1
    releaseVisitedMap<false>(std::move(visited));
    return ~distance;
}


void output_stats_vamana(RawVectorData *vectors, int k, RawVectorData *queries, int *result, Index *raw_index) {
    (void) vectors; (void) k; (void) result;
    size_t total_hops = 0;
    auto index = static_cast<VamanaIndex *>(raw_index);
    auto hop_list = index->hop_list;
    for (int h : hop_list) total_hops += h;
    std::cout << "[STATS] avg num vertices explored: " << std::setprecision(6) << (double) total_hops / queries->length << std::endl;

    std::stringstream out_fn;
    out_fn << "hopdist_" << out_fn_vamana(index, nullptr);
    std::ofstream f(out_fn.str(), std::ios::binary);
    f.write((const char *) hop_list.data(), hop_list.size() * sizeof hop_list[0]);

    std::vector<OutputData> vec;
    vec.resize(index->maxVertices);

    {
        ScopedTimer timer("basic info");
        for (int i = 0; i < (int) vec.size(); i++) {
            VertexPtr v = VertexPtr(i);
            vec[i].id = i;
            vec[i].level = 0;
            vec[i].degree = index->get(v).numNeighbors;
            vec[i].distance = index->distance(v);
        }
    }

    out_fn.clear(); out_fn.str(std::string());
    out_fn << "graph_" << out_fn_vamana(index, nullptr);
    f.close();
    f.clear();
    f.open(out_fn.str(), std::ios::binary);
    f.write((const char *) vec.data(), vec.size() * sizeof vec[0]);
}
