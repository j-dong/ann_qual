#pragma once

#include <memory>

class Index {
public:
    virtual ~Index()=0;
};
inline Index::~Index() {}

struct RawVectorData;

typedef std::unique_ptr<Index> preprocess_decl(bool is_l2, RawVectorData *vectors, RawVectorData *learn);

typedef int compute_decl(RawVectorData *vectors, int k, float *query, int *result, Index *index);
