#ifndef MAT_MUL
#define MAT_MUL

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../utils.h"

__global__ void matmul_cuda_1(float* a, float* b, float* c) {
    
}

void matmul_cublas(float* out, float* a, float* b,
                   bool a_trans, bool b_trans,
                   int row, int inner, int col) {
    // AB
    if (!a_trans && !b_trans) {
        CUBLAS_CHECK(cublasSgemm_v2(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            col, row, inner, &ALPHA,
            b, col, a, inner, &BETA,
            out, col
        ));
    }
    // AB^T
    else if (!a_trans && b_trans) {
        CUBLAS_CHECK(cublasSgemm_v2(
            cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            col, row, inner, &ALPHA,
            b, inner, a, col, &BETA,
            out, col
        ));
    }
}

#endif