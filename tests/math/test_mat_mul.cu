#include <iostream>
#include <cublas_v2.h>
#include "../../src/utils.h"

void test1() {

}

void test2() {
    
}

void test_cublas1() {
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    
    const float alpha = 1.0;
    const float beta = 0.0;
    const int m = 2;
    const int n = 3;
    const int k = 2;
    
    auto a = new float[m * k];
    auto b = new float[k * n];
    auto c = new float[m * n];

    fill_increment<float>(a, m * k);
    fill_increment<float>(b, k * n);
    // A: m x k
    // B: k x n

    // B^T: n * k
    // A^T: k * m

    // A @ B = C
    // B^T @ A^T = C^T -- row-major
    // B @ A = C -- column major
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * m * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * k * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * m * n));

    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    // input leading dims as if it was column-major...
    // switch operands & transpose second matrix
    CUBLAS_CHECK(cublasSgemm_v2(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha,
        d_b, n, d_a, k,
        &beta, d_c, n
    ));
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    print_matrix(m, n, c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;
    
    cublasDestroy(handle);
    cudaDeviceReset();
}

int main(int argc, char** argv) {
    // test1();
    test_cublas1();
}
