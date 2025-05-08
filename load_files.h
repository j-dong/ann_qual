#include <cstdint>

struct RawVectorData {
    void *filestart;
    std::size_t filesize;
    float *vec;
    inline float &at(int i, int j) { return vec[1 + j + i * stride()]; }
    int dim;
    inline int stride() const { return dim + 1; }
    std::size_t length;
};

struct IntVectorData {
    void *filestart;
    std::size_t filesize;
    int *vec;
    inline int &at(int i, int j) { return vec[1 + j + i * stride()]; }
    int dim;
    inline int stride() const { return dim + 1; }
    std::size_t length;
};

extern RawVectorData data_base;
extern RawVectorData data_query;
extern RawVectorData data_learn;
extern IntVectorData data_ground;

void load_files();
void close_files();
