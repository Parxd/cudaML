#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// N blocks, each with 1 thread
__global__ void vecadd_1(float *a, float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// one block with N threads
__global__ void vecadd_2(float* a, float* b, float* c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

// N blocks with M threads (assumes N is a "friendly" multiple)
__global__ void vecadd_3(float* a, float* b, float* c) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    c[idx] = a[idx] + b[idx];
}

// N may not be a "friendly" multiple
// N divided by number of threads may not result in an integer
// must check if index is within bounds of N
__global__ void vecadd_4(float* a, float* b, float* c, int size) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
