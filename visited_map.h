#pragma once

#include <cstring>
#include <vector>

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
        vec = o.vec; o.vec = nullptr;
        return *this;
    }
    ~VisitedMap() { delete[] vec; }

    bool operator[](size_t i) const { return vec[i] == tag; }
    void set(size_t i) { vec[i] = tag; }

    void reset() { tag++; }

    static VisitedMap takeFromPool(std::vector<VisitedMap> &pool, size_t maxVertices) {
        if (pool.size() == 0) {
            pool.emplace_back(maxVertices);
        }
        VisitedMap ret = std::move(pool.back());
        pool.pop_back();
        if (!ret.cleared) {
            memset(ret.vec, 0, maxVertices);
            ret.tag = 0;
        }
        ret.tag++;
        return ret;
    }
};
