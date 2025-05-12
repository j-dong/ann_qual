#include "kernel_hnsw.h"
#include "kernel_utils.h"
#include "load_files.h"
#include "timer.h"
#include "simd_utils.h"

#ifndef NOMINMAX
# define NOMINMAX 1
#endif

#include <cstring>
#include <vector>
#include <queue>
#include <random>
#include <stdexcept>
#include <iostream>

struct HNSWIndex;
void *resolveVertex(HNSWIndex *, uint32_t);

namespace {
struct Ptr {
    uint32_t i;

    Ptr() = default;
    explicit Ptr(uint32_t i) : i(i) {}
    Ptr(nullptr_t) : i(UINT32_MAX) {}
    explicit operator bool() const { return i != UINT32_MAX; }
};
struct VertexPtr : Ptr { using Ptr::Ptr; };
struct LinkPtr : Ptr { using Ptr::Ptr; };

struct Vertex;

struct Link {
    VertexPtr vertex;
    LinkPtr next;

    Link(VertexPtr vertex, LinkPtr next) : vertex(vertex), next(next) {}
};

struct Vertex {
    LinkPtr neighbors = nullptr;
    VertexPtr below = nullptr;
    int numNeighbors = 0;
    int id;
    float *data;

    Vertex(int id, float *data) : id(id), data(data) {}
};

struct PQElement {
    float dist;
    VertexPtr vertex;

    PQElement(float dist, VertexPtr vertex) : dist(dist), vertex(vertex) {}
};

struct CompMaxDist {
    bool operator()(const PQElement &a, const PQElement &b) {
        return a.dist < b.dist;
    }
};
struct CompMinDist {
    bool operator()(const PQElement &a, const PQElement &b) {
        return a.dist > b.dist;
    }
};
using CompareDistance = CompMaxDist;

enum class Lowering {
    SameLevel,
    Lower,
};

template<typename Comp>
struct my_heap : std::priority_queue<PQElement, std::vector<PQElement>, Comp> {
    using Base = std::priority_queue<PQElement, std::vector<PQElement>, Comp>;
    template<typename Comp2>
    friend struct my_heap;

    const std::vector<PQElement> &elements() { return this->c; }
    const std::vector<PQElement> move_elements() && { return std::move(this->c); }

    my_heap() : Base::priority_queue() {}
    my_heap(std::vector<PQElement> &&vec) : Base::priority_queue(Comp(), vec) {}

    void lower(HNSWIndex *idx) {
        for (PQElement &x : this->c) {
            Vertex *v = (Vertex *) resolveVertex(idx, x.vertex.i);
            x.vertex = v->below;
        }
    }
    template<typename Comp2>
    my_heap(const my_heap<Comp2> &other, Lowering lower, HNSWIndex *idx) : Base(Comp(), other.c) {
        switch (lower) {
        case Lowering::Lower:
            this->lower(idx);
            break;
        case Lowering::SameLevel:
            break;
        }
    }
};

struct max_heap : my_heap<CompMaxDist> {
    using my_heap<CompMaxDist>::my_heap;
};
struct min_heap : my_heap<CompMinDist> {
    using my_heap<CompMinDist>::my_heap;
};

struct VisitedMap {
    char *vec = nullptr;
    char tag;
    bool cleared = false;

    VisitedMap(size_t size) {
        vec = new char[size];
    }
    VisitedMap(VisitedMap &&o) : vec(o.vec) {
        o.vec = nullptr;
    }
    VisitedMap &operator=(VisitedMap &&o) {
        if (&o == this) return *this;
        if (vec) { delete vec; vec = nullptr; }
        std::swap(vec, o.vec);
    }
    ~VisitedMap() { delete[] vec; }
};
}

struct HNSWIndex : Index {
    int dim;

    std::vector<Vertex> vertices;
    std::vector<Link> links;
    LinkPtr freeList = nullptr;
    int maxVertices;
    int maxDegree;
    double m_L;
    int efConstruction;

    VertexPtr entry = nullptr;
    int max_level = -1;

    std::mt19937 rng;
    std::vector<VisitedMap> visited_pool;

    HNSWIndex(int dim, int maxVertices, int maxDegree, int efConstruction)
            : dim(dim), maxVertices(maxVertices), maxDegree(maxDegree), m_L(1.0 / std::log((double) maxDegree)), efConstruction(efConstruction) {
        rng.seed(0xdeadbeef);
        // extremely conservative bound
        vertices.reserve(2 * maxVertices);
        links.reserve(2 * maxVertices * maxDegree);
        // initialize visited maps
        // if multi-threaded, we should initialize one for each thread
        for (int i = 0; i < 1; i++) {
            visited_pool.emplace_back(maxVertices);
        }
    }

    int getRandomLevel() {
        double rand = std::generate_canonical<double, 50>(rng);
        if (rand == 0.0) rand = std::numeric_limits<double>::epsilon();
        double l = -std::log(rand) * m_L;
        if (l > 40.0) return 40;
        return (int) l;
    }

    Vertex &get(VertexPtr p) { return vertices[p.i]; }
    Vertex &get(Vertex *v) const { return *v; }
    Link &get(LinkPtr p) { return links[p.i]; }
    VertexPtr makeVertex(int id, float *data) {
        VertexPtr p((uint32_t) vertices.size());
        vertices.emplace_back(id, data);
        return p;
    }
    LinkPtr makeLink(VertexPtr vertex, LinkPtr next) {
        if (freeList) {
            LinkPtr ret = freeList;
            freeList = get(ret).next;
            get(ret).vertex = vertex;
            get(ret).next = next;
            return ret;
        }
        LinkPtr p((uint32_t) links.size());
        links.emplace_back(vertex, next);
        return p;
    }

    void release(LinkPtr *p) { release(*p); *p = nullptr; }
    void release(LinkPtr p) {
        LinkPtr cur = p;
        while (cur) {
            Link &l = get(cur);
            l.vertex = nullptr;
            if (!l.next) {
                l.next = freeList;
                break;
            } else {
                cur = l.next;
            }
        }
        freeList = p;
    }

    VisitedMap takeVisitedMap() {
        if (visited_pool.size() == 0) {
            visited_pool.emplace_back(maxVertices);
        }
        VisitedMap ret = std::move(visited_pool.back());
        if (!ret.cleared) {
            memset(ret.vec, 0, maxVertices);
            ret.tag = 0;
        }
        ret.tag++;
        visited_pool.pop_back();
        return ret;
    }

    void releaseVisitedMap(VisitedMap &&map) {
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
    max_heap searchLayer(float *data, max_heap ep, int ef);
    std::vector<PQElement> selectNeighbors(std::vector<PQElement> candidates, int M);
    std::vector<PQElement> query(float *data, int k, int ef);
};

inline void *resolveVertex(HNSWIndex *idx, uint32_t p) {
    return (void *) &idx->get(VertexPtr(p));
}



std::unique_ptr<Index> preprocess_ann_hnsw(bool is_l2, RawVectorData *vectors, RawVectorData *learn) {
    if (!is_l2) {
        throw std::runtime_error("we only support L2 for HNSW");
    }
    (void) learn;
    auto ret = std::make_unique<HNSWIndex>(vectors->dim, vectors->length, 16, 64);
    for (int i = 0; i < vectors->length; i++) {
        if (i % 1000 == 0) {
            std::cout << "insert progress: " << i << "/" << vectors->length << std::endl;
        }
        float *data = &vectors->at(i, 0);
        ret->insert(i, data);
    }
    return ret;
}

int compute_ann_hnsw(RawVectorData *vectors, int k, float *query, int *result, Index *raw_index) {
    HNSWIndex *index = (HNSWIndex *) raw_index;
    auto vec = index->query(query, k, std::max(10, k));
    for (int i = 0; i < (int) vec.size(); i++) {
        float *data = index->get(vec[i].vertex).data;
        result[i] = (data - &vectors->at(i, 0)) / (vectors->dim + 1);
    }
    return vec.size();
}



void HNSWIndex::insert(int id, float *data) {
    max_heap ep;
    Vertex *above = nullptr;
    VertexPtr root = nullptr;
    int cur_level = max_level;
    int ins_level = getRandomLevel();
    if (ins_level > max_level) {
        for (cur_level = ins_level; cur_level > max_level; cur_level--) {
            VertexPtr q_ptr = makeVertex(id, data);
            if ((!root) != (above == nullptr)) { [[unlikely]]
                throw std::runtime_error("root existing should match above existing");
            }
            if (!root) root = q_ptr;
            if (above) above->below = q_ptr;
            above = &get(q_ptr);
        }
    } else {
        if (!entry) { [[unlikely]]
            throw std::runtime_error("assertion failure:"
                " entry should exist if ins_level <= max_level");
        }
    }
    if (entry) pushq(ep, entry, data);
    for (; cur_level > ins_level; cur_level--) {
        ep = searchLayer(data, std::move(ep), 1);
        ep.lower(this);
    }
    for (; cur_level >= 0; cur_level--) {
        VertexPtr q_ptr = makeVertex(id, data);
        if (above) above->below = q_ptr;
        Vertex &q = get(q_ptr);
        ep = searchLayer(data, std::move(ep), efConstruction);
        int M = cur_level == 0 ? 2 * maxDegree : maxDegree;
        auto neighbors = selectNeighbors(ep.elements(), M);
        for (auto &n : neighbors) {
            q.neighbors = makeLink(n.vertex, q.neighbors);
        }
        q.numNeighbors = neighbors.size();
        for (auto &n : neighbors) {
            Vertex &e = get(n.vertex);
            if (e.numNeighbors < M) {
                e.neighbors = makeLink(q_ptr, e.neighbors);
                e.numNeighbors++;
                continue;
            }
            std::vector<PQElement> temp;
            for (LinkPtr l = e.neighbors; l; l = get(l).next) {
                VertexPtr v = get(l).vertex;
                temp.emplace_back(computeDistance(get(v).data, e.data), v);
            }
            temp.emplace_back(n.dist, q_ptr);
            auto new_neighbors = selectNeighbors(std::move(temp), M);
            auto it = new_neighbors.rbegin();
            auto it_end = new_neighbors.rend();
            LinkPtr *l = &e.neighbors;
            for (; it != it_end; ++it, l = &get(*l).next) {
                get(*l).vertex = it->vertex;
            }
            release(l);
            e.numNeighbors = new_neighbors.size();
        }
        above = &q;
        ep.lower(this);
    }
    if (ins_level > max_level) {
        max_level = ins_level;
        if (!root) [[unlikely]] throw std::runtime_error("root should exist");
        entry = root;
    }
}

max_heap HNSWIndex::searchLayer(float *data, max_heap ep, int ef) {
    VisitedMap v = takeVisitedMap();
    auto *visited = v.vec;
    auto visit_tag = v.tag;

    min_heap candidates(ep, Lowering::SameLevel, this);
    max_heap nearest = std::move(ep);
    float farthest_dist = nearest.top().dist;

    while (!candidates.empty()) {
        PQElement cur = candidates.top();
        candidates.pop();
        if (cur.dist > farthest_dist) {
            break;
        }
        for (LinkPtr l = get(cur.vertex).neighbors; l; l = get(l).next) {
            Link &link = get(l);
            VertexPtr e = link.vertex;
            int id = get(e).id;
            if (visited[id]) continue;
            visited[id] = visit_tag;
            float dist = computeDistance(get(e).data, data);
            if (dist < farthest_dist || candidates.size() < (size_t) ef) {
                pushq(candidates, e, dist);
                if (nearest.size() >= (size_t) ef) {
                    nearest.pop();
                }
                pushq(nearest, e, dist);
            }
        }
    }

    return nearest;
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
        }
    }
    candidates.erase(candidates.begin() + o, candidates.end());
    return candidates;
}

std::vector<PQElement> HNSWIndex::query(float *data, int k, int ef) {
    max_heap ep;
    pushq(ep, entry, data);
    int cur_level = max_level;
    for (; cur_level > 0; cur_level--) {
        ep = searchLayer(data, std::move(ep), 1);
        ep.lower(this);
    }
    max_heap nearest = searchLayer(data, std::move(ep), ef);
    while (nearest.size() > (size_t) k) {
        nearest.pop();
    }
    std::vector<PQElement> ret = std::move(nearest).move_elements();
    static_assert(std::is_same<decltype(nearest)::value_compare, CompareDistance>::value, "must sort using same comparator");
    std::sort_heap(ret.begin(), ret.end(), CompareDistance());
    return ret;
}
