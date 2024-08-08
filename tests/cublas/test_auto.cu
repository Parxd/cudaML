#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/utils.h"
#include "../../src/cublas/add.cu"
#include "../../src/cublas/sum.cu"

int main(int argc, char** argv) { 
    CUBLAS_CHECK(cublasCreate_v2(&cublas_handle));  // don't declare cublas_handle; create the static instance from utils.h
    // cublasSetPointerMode_v2(cublas_handle, CUBLAS_POINTER_MODE_DEVICE); 
    int size = 10;
    int streams = 2;
    cudaStream_t stream_arr[streams];

    CUDA_CHECK(cudaStreamCreate(&stream_arr[0]));
    CUDA_CHECK(cudaStreamCreate(&stream_arr[1]));
    // TODO: wrap the rest of these in CUDA_CHECK macro
    float *a, *b, *c, *d;
    cudaMallocHost((void**)&a, sizeof(float) * size);
    cudaMallocHost((void**)&b, sizeof(float) * size);
    cudaMallocHost((void**)&c, sizeof(float) * size);
    cudaMallocHost((void**)&d, sizeof(float));
    fill_increment<float>(a, size);
    fill_increment<float>(b, size);

    float *d_a, *d_b, *d_c, *d_d;
    cudaMallocAsync((void**)&d_a, sizeof(float) * size, stream_arr[0]);
    cudaMallocAsync((void**)&d_b, sizeof(float) * size, stream_arr[0]);
    cudaMallocAsync((void**)&d_c, sizeof(float) * size, stream_arr[1]);
    // cudaMallocAsync((void**)&d_d, sizeof(float), stream_arr[1]);
    cudaMemcpyAsync(d_a, a, sizeof(float) * size, cudaMemcpyHostToDevice, stream_arr[0]);
    cudaMemcpyAsync(d_b, b, sizeof(float) * size, cudaMemcpyHostToDevice, stream_arr[0]);

    // cudaDeviceSynchronize();  // do we need this?

    // "forward" pass
    add_cublas(d_c, d_a, d_b, 1, size);
    sum_cublas(d, d_c, size);
    
    // cudaMemcpy(d, d_d, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << *d << std::endl;  // 110 for size=10
    
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // cudaFree(d_d);
    CUBLAS_CHECK(cublasDestroy_v2(cublas_handle));
    return 0;
}