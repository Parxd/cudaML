#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils.h"

static cublasHandle_t cublas_handle;

__global__ void matmul_1(float* a, float* b, float* c) {
    
}

void matmul_forward(float *out, float* a, float* b,
                    int row, int inner, int col) {
    CUBLAS_CHECK(cublasSgemm_v2(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        col, row, inner, &ALPHA,
        b, col, a, inner, &BETA,
        out, col
    ));
}
