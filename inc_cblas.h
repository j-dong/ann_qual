#if __has_include(<mkl.h>)
# include <mkl.h>
# define BLAS_MKL
#else
# include <cblas.h>
#endif
