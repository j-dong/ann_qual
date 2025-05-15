#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>

struct Ptr {
    uint32_t i;

    Ptr() = default;
    explicit Ptr(uint32_t i) : i(i) {}
    Ptr(nullptr_t) : i(UINT32_MAX) {}
    explicit operator bool() const { return i != UINT32_MAX; }

    bool operator==(const Ptr &o) const { return i == o.i; }
};

struct VertexPtr : Ptr { using Ptr::Ptr; };

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

template<typename Comp>
struct vertex_heap : std::priority_queue<PQElement, std::vector<PQElement>, Comp> {
    using Base = std::priority_queue<PQElement, std::vector<PQElement>, Comp>;
    template<typename Comp2>
    friend struct vertex_heap;

    const std::vector<PQElement> &elements() { return this->c; }
    const std::vector<PQElement> move_elements() && { return std::move(this->c); }

    vertex_heap() : Base::priority_queue() {}
    explicit vertex_heap(std::vector<PQElement> &&vec) : Base::priority_queue(Comp(), vec) {}

    template<typename Comp2>
    explicit vertex_heap(const vertex_heap<Comp2> &other) : Base(Comp(), other.c) {}
};

