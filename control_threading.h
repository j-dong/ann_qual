#pragma once

#include <thread>

#include "inc_cblas.h"

#if __has_include(<blis.h>)
# include <blis.h>
# define BLAS_BLIS
#elif __has_include(<mkl.h>)
# define BLAS_MKL
#endif

inline void enable_blas_threading() {
#if defined(BLAS_BLIS)
    bli_thread_set_num_threads(std::thread::hardware_concurrency());
#elif defined(BLAS_MKL)
    mkl_set_num_threads(std::thread::hardware_concurrency());
#else
    openblas_set_num_threads(std::thread::hardware_concurrency());
#endif
}

inline void disable_blas_threading() {
#if defined(BLAS_BLIS)
    bli_thread_set_num_threads(1);
#elif defined(BLAS_MKL)
    mkl_set_num_threads(1);
#else
    openblas_set_num_threads(1);
#endif
}
