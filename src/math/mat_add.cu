#include <cstdio>
#include <cuda_runtime.h>

// as pointed in docs, matrix addition doesn't really need 
__global__ void matadd_1(float* a, float* b, float* c) {
    
}

__global__ void matadd_2(float* a, float* b, float* c, int N, int M) {
    int idx = threadIdx.x * N + threadIdx.y;
    // don't really need to pass N and M into this kernel, can just use blockDim values, but equivalent either way
    // int idx = threadIdx.x * blockDim.x + threadIdx.y
    c[idx] = a[idx] + b[idx];
}
