#include <iostream>
#include <cublas_v2.h>
#include "../../src/utils.h"
#include "../../src/math/mat_mul.cu"

void test1() {

}

void test2() {
    
}

void test_cublas1() {
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    
    const float alpha = 1.0;
    const float beta = 0.0;
    const int m = 4;
    const int n = 3;
    const int k = 5;
    
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
    // switch & transpose operands
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

void test_cublas2() {
    cublasCreate(&cublas_handle);
    // want to test actual function
    auto a = new float[3 * 5];
    auto b = new float[5 * 4];
    auto c = new float[3 * 4];

    fill_increment<float>(a, 15);
    fill_increment<float>(b, 20);
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * 15));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * 20));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * 12));

    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * 15, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * 20, cudaMemcpyHostToDevice));

    matmul_forward(d_c, d_a, d_b, 3, 5, 4);

    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * 12, cudaMemcpyDeviceToHost));
    
    print_matrix(3, 4, c, 4);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(cublas_handle);
}

int main(int argc, char** argv) {
    // test1();
    // test_cublas1();
    test_cublas2();
}
