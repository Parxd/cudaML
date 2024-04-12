#include "../../src/utils.h"
#include "../../src/math/vec_add.cu"

static int N = 18;
static int THREADS_PER_BLOCK = 4;


int main(int argc, char** argv) {
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
    
    // vecadd_1<<<N, 1>>>(d_a, d_b, d_c);
    // vecadd_2<<<1, N>>>(d_a, d_b, d_c);
    // vecadd_3<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    vecadd_4<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    
    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%d\n", c[i]);
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
