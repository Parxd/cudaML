#include <iostream>
#include <cublas_v2.h>
#include "../../include/utils.h"
#include "../../src/cublas/mat_mul.cu"

void test_cuda1() {

}

void test_cublas1() {
    // testing standard gemm
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
    // for (int i = 0; i < n * m; ++i) {
    //     std::cout << c[i] << "\t";
    // }
    // std::cout << std::endl;

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
    // testing standard gemm
    cublasCreate(&cublas_handle);
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

    matmul_cublas(d_c, d_a, d_b, false, false, 3, 5, 4);

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

void test_cublas3() {
    // testing right-side transpose gemm
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);
    
    const float alpha = 1.0;
    const float beta = 0.0;
    
    auto a = new float[1 * 2];
    auto b = new float[3 * 2];
    auto c = new float[1 * 3];

    fill_increment<float>(a, 1 * 2);
    fill_increment<float>(b, 3 * 2);

    // print_matrix(1, 2, a, 2);
    // print_matrix(3, 2, b, 2);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * 1 * 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * 3 * 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * 1 * 3));
    
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * 1 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * 3 * 2, cudaMemcpyHostToDevice));
    
    CUBLAS_CHECK(cublasSgemm_v2(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        3, 1, 2, &alpha, d_b, 2, d_a, 3, &beta, d_c, 3
    ));
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * 1 * 3, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 3; i++) {
    //     std::cout << c[i] << " ";
    // }
    // std::cout << std::endl;
    print_matrix(1, 3, c, 3);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;
    
    cublasDestroy(handle);
    cudaDeviceReset();
}

void test_cublas4() {
    // testing right-side transpose gemm
    cublasCreate(&cublas_handle);
    
    auto a = new float[1 * 2];
    auto b = new float[3 * 2];
    auto c = new float[1 * 3];

    fill_increment<float>(a, 1 * 2);
    fill_increment<float>(b, 3 * 2);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * 1 * 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * 3 * 2));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * 1 * 3));
    
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * 1 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * 3 * 2, cudaMemcpyHostToDevice));
    
    matmul_cublas(d_c, d_a, d_b, false, true, 1, 2, 3);

    // CUBLAS_CHECK(cublasSgemm_v2(
    //     handle,
    //     CUBLAS_OP_T, CUBLAS_OP_N,
    //     3, 1, 2, &ALPHA, d_b, 2, d_a, 3, &BETA, d_c, 3
    // ));

    CUDA_CHECK(cudaMemcpy(c, d_c, sizeof(float) * 3 * 1, cudaMemcpyDeviceToHost));

    print_matrix(1, 3, c, 3);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;
    
    cublasDestroy(cublas_handle);
    cudaDeviceReset();
}

int main(int argc, char** argv) {
    // test_cuda1();
    // test_cublas1();
    // test_cublas2();
    // test_cublas3();
    test_cublas4();
}
