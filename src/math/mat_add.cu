#include <cstdio>
#include <cuda_runtime.h>

// as pointed in docs, matrix addition really doesn't need 2D blocks
__global__ void matadd_1(float* a, float* b, float* c, int N, int M) {
    int idx = threadIdx.x * N + threadIdx.y;
    // don't really need to pass N and M into this kernel, can just use blockDim values, but equivalent either way
    // int idx = threadIdx.x * blockDim.x + threadIdx.y
    c[idx] = a[idx] + b[idx];
}

// what if our matrix exceeds the max 2D block size of 1024 threads?
// we need to split into more blocks like vecadd_4()
__global__ void matadd_2(float* a, float* b, float* c) {
    
}
