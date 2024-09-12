#include "cutlass/gemm/device/gemm.h"
#include "../include/tensorimpl.cuh"
#include "../include/math/matmul.cuh"
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    using dtype = float;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // TensorImpl<dtype> A(3, 5);
    // TensorImpl<dtype> B(5, 6);
    // auto C = mm(A, B, stream);
    // C.print_tensor(stream);

    int M = 3;
    int N = 5;
    int K = 6;
    int byte_size_a = sizeof(float) * M * K;
    int byte_size_b = sizeof(float) * K * N;
    int byte_size_c = sizeof(float) * M * N;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    fill_increment<float>(A, M * K);
    fill_increment<float>(B, K * N);
    float *d_A;
    float *d_B;
    float *d_C;
    cudaMallocAsync((void**)&d_A, byte_size_a, stream);
    cudaMallocAsync((void**)&d_B, byte_size_b, stream);
    cudaMallocAsync((void**)&d_C, byte_size_c, stream);
    cudaMemcpyAsync(d_A, A, byte_size_a, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, byte_size_b, cudaMemcpyHostToDevice, stream);
    
    cudaStreamDestroy(stream);
}
