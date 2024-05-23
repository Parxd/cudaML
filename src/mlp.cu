#include <cublas_v2.h>
#include "math/mat_add.cu"
#include "math/mat_mul.cu"

int main(int argc, char** argv) {
    cublasCreate(&cublas_handle);
    int config[5] = {3, 5, 10, 5, 2};
    int batch_size = 32;

    float* w1 = (float*)mallocCheck(sizeof(float) * config[0] * config[1]);  // (5, 3)
    float* w2 = (float*)mallocCheck(sizeof(float) * config[1] * config[2]);  // (10, 5)
    float* w3 = (float*)mallocCheck(sizeof(float) * config[2] * config[3]);  // (5, 10)
    float* w4 = (float*)mallocCheck(sizeof(float) * config[3] * config[4]);  // (2, 5)
    float* b1 = (float*)mallocCheck(sizeof(float) * config[1]);  // (5, 1)
    float* b2 = (float*)mallocCheck(sizeof(float) * config[2]);  // (10, 1)
    float* b3 = (float*)mallocCheck(sizeof(float) * config[3]);  // (5, 1)
    float* b4 = (float*)mallocCheck(sizeof(float) * config[4]);  // (2, 1)

    float* input = (float*)mallocCheck(sizeof(float) * batch_size * config[0]);  //  input shape (32, 3)
    // y = (x @ W^T) + b
    float* f1 = (float*)mallocCheck(sizeof(float) * batch_size * config[1]);
    float* f2 = (float*)mallocCheck(sizeof(float) * batch_size * config[2]);
    float* f3 = (float*)mallocCheck(sizeof(float) * batch_size * config[3]);
    float* f4 = (float*)mallocCheck(sizeof(float) * batch_size * config[4]);

    float* d_w1, *d_w2, *d_w3, *d_w4, *d_b1, *d_b2, *d_b3, *d_b4, *d_input, *d_f1, *d_f2, *d_f3, *d_f4;
    CUDA_CHECK(cudaMalloc((void**)&d_w1, sizeof(float) * config[0] * config[1]));
    CUDA_CHECK(cudaMalloc((void**)&d_w2, sizeof(float) * config[1] * config[2]));
    CUDA_CHECK(cudaMalloc((void**)&d_w3, sizeof(float) * config[2] * config[3]));
    CUDA_CHECK(cudaMalloc((void**)&d_w4, sizeof(float) * config[3] * config[4]));
    CUDA_CHECK(cudaMalloc((void**)&d_b1, sizeof(float) * config[1]));
    CUDA_CHECK(cudaMalloc((void**)&d_b2, sizeof(float) * config[2]));
    CUDA_CHECK(cudaMalloc((void**)&d_b3, sizeof(float) * config[3]));
    CUDA_CHECK(cudaMalloc((void**)&d_b4, sizeof(float) * config[4]));
    CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(float) * batch_size * config[0]));
    CUDA_CHECK(cudaMalloc((void**)&d_f1, sizeof(float) * batch_size * config[1]));
    CUDA_CHECK(cudaMalloc((void**)&d_f2, sizeof(float) * batch_size * config[2]));
    CUDA_CHECK(cudaMalloc((void**)&d_f3, sizeof(float) * batch_size * config[3]));
    CUDA_CHECK(cudaMalloc((void**)&d_f4, sizeof(float) * batch_size * config[4]));

    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_w3));
    CUDA_CHECK(cudaFree(d_w4));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_b3));
    CUDA_CHECK(cudaFree(d_b4));
    CUDA_CHECK(cudaFree(d_f1));
    CUDA_CHECK(cudaFree(d_f2));
    CUDA_CHECK(cudaFree(d_f3));
    CUDA_CHECK(cudaFree(d_f4));
    cublasDestroy(cublas_handle);
    free(w1);
    free(w2);
    free(w3);
    free(w4);
    free(b1);
    free(b2);
    free(b3);
    free(b4);
    return 0;
}
