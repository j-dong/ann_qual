#include <cstdint>

struct RawVectorData {
    void *filestart;
    size_t filesize;
    float *vec;
    inline float &at(int i, int j) { return vec[1 + j + i * stride()]; }
    int dim;
    inline int stride() const { return dim + 1; }
    size_t length;
};

extern RawVectorData data_base;
extern RawVectorData data_query;
extern RawVectorData data_learn;

void load_files();
void close_files();
