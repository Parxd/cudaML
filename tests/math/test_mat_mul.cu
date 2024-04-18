#include <iostream>
#include <cublas_v2.h>
#include "../../src/utils.h"

void test1() {

}

void test2() {
    
}

void test_cublas1() {
    cublasHandle_t handle = NULL;
    cudaStream_t stream = NULL;
    cublasCreate(&handle);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);
    
    const float alpha = 1.0;
    const float beta = 0.0;
    const int m = 2;
    const int n = 2;
    const int k = 2;

    auto a = new float[m * k];
    auto b = new float[k * n];
    auto c = new float[m * n];

    fill_increment<float>(a, m * k);
    fill_increment<float>(b, k * n);

    float *d_a, *d_b, *d_c;
    cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float) * m * k);
    cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float) * k * n);
    cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float) * m * n);

    cudaMemcpy(d_a, a, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    
    cublasSgemm_v2(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha,
        d_a, k, d_b, n,
        &beta, d_c, n
    );
    cudaMemcpy(c, d_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    print_matrix(m, n, c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;
}

int main(int argc, char** argv) {
    test1();
    test_cublas1();
}
