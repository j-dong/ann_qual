#pragma once

#include <cblas.h>
#include <thread>

#if defined(BLAS_BLIS)
# include <blis.h>
#endif

inline void enable_blas_threading() {
#if defined(BLAS_BLIS)
    bli_thread_set_num_threads(std::thread::hardware_concurrency());
#elif defined(BLAS_OPENBLAS)
    openblas_set_num_threads(std::thread::hardware_concurrency());
#endif
}

inline void disable_blas_threading() {
#if defined(BLAS_BLIS)
    bli_thread_set_num_threads(1);
#elif defined(BLAS_OPENBLAS)
    openblas_set_num_threads(1);
#endif
}
