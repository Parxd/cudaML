#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// static int N = 2048;
// static int THREADS_PER_BLOCK = 512;
static int N = 18;
static int THREADS_PER_BLOCK = 4;

// N blocks, each with 1 thread
__global__ void kernel_1(float *a, float* b, float* c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// one block with N threads
__global__ void kernel_2(float* a, float* b, float* c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

// N blocks with M threads (assumes N is a "friendly" multiple)
__global__ void kernel_3(float* a, float* b, float* c) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    c[idx] = a[idx] + b[idx];
}

// N may not be a "friendly" multiple
// N divided by number of threads may not result in an integer
// must check if index is within bounds of N
__global__ void kernel_4(float* a, float* b, float* c, int size) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void fill_ones(float* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = 1.0;
    }
}

int main() {
    int byte_size = N * sizeof(float);

    float *a, *b, *c;
    a = (float*)malloc(byte_size);
    b = (float*)malloc(byte_size);
    c = (float*)malloc(byte_size);
    
    fill_ones(a, N);
    fill_ones(b, N);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, byte_size);
    cudaMalloc((void**)&d_b, byte_size);
    cudaMalloc((void**)&d_c, byte_size);
    
    cudaMemcpy(d_a, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, byte_size, cudaMemcpyHostToDevice);
    
    // kernel_1<<<N, 1>>>(d_a, d_b, d_c);
    // kernel_2<<<1, N>>>(d_a, d_b, d_c);
    // kernel_3<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // if N is an unfriendly multiple--use integer divison
    // for ex. if N = 18 & threads = 4, (18 + 4 - 1) / 4 = 5 --> launches 5 blocks with 4 threads each
    kernel_4<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d %s %.2f\n", i, "->", c[i]);
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
