#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm.h"

#include <cuda_runtime.h>
#include "../include/tensorimpl.cuh"
#include "../include/utils.h"

using namespace cutlass;
using namespace cute;

template <typename TensorType>
bool is_device_tensor(const TensorType& tensor) {
    return is_device_pointer(tensor.data());
}

int main(int argc, char* argv[]) {
    using dtype = float;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
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
    using Gemm = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm86
    >;
    Gemm gemm;
    Gemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1.0f, 0.0f}
    );
    gemm(args, stream);
    cudaMemcpyAsync(C, d_C, byte_size_c, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    for (int i = 0; i < M * N; ++i) {
        std::cout << C[i] << " ";
    }
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    cudaStreamDestroy(stream);
    std::cout << std::endl;
} 
