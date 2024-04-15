#include "../../src/utils.h"
#include "../../src/math/mat_add.cu"

void test1() {
    int N = 2;
    int M = 3;
    int byte_size = N * M * sizeof(float);

    float *a, *b, *c;
    a = (float*)malloc(byte_size);
    b = (float*)malloc(byte_size);
    c = (float*)malloc(byte_size);
    
    fill_ones(a, N * M);
    fill_ones(b, N * M);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, byte_size);
    cudaMalloc((void**)&d_b, byte_size);
    cudaMalloc((void**)&d_c, byte_size);
    
    cudaMemcpy(d_a, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, byte_size, cudaMemcpyHostToDevice);

    // with 1D block:
    // vecadd_2<<<1, N * M>>>(d_a, d_b, d_c);

    // with 2D block:
    // need 1 block of shape 2 x 3
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(N, M, 1);
    matadd_1<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, M);

    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    print_matrix(c, N, M);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void test2() {
    int N = 41;
    int M = 37;
    int byte_size = N * M * sizeof(float);

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

    dim3 gridDim(CEIL_DIV(N, MAX_THREADS), CEIL_DIV(M, MAX_THREADS));
    dim3 blockDim(MAX_THREADS, MAX_THREADS, 1);
    matadd_2<<<gridDim, blockDim>>>(d_a, d_b, d_c);
}

int main(int argc, char** argv) {
    // test1();
    test2();
}
