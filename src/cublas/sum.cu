#ifndef SUM
#define SUM

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils.h"

void sum_cublas(float* sum, float* x, int size) {
    CUBLAS_CHECK(cublasSasum(cublas_handle, size, x, 1, sum));
}

#endif