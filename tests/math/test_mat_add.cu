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
    
    fill_increment(a, N * M);
    fill_increment(b, N * M);

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

    print_matrix(N, M, c, M);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void test2() {
    // tile quantization
    int N = 35;
    int M = 33;
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

    dim3 gridDim(CEIL_DIV(N, MAX_THREADS), CEIL_DIV(M, MAX_THREADS), 1);
    dim3 blockDim(MAX_THREADS, MAX_THREADS, 1);
    matadd_2<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    print_matrix(N, M, c, M);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}

void test_cublas1() {
    // testing out-of-place geam
    cublasCreate(&cublas_handle);

    int rows = 15;
    int cols = 17;
    int byte_size = rows * cols * sizeof(float);

    auto a = new float[rows * cols];
    auto b = new float[rows * cols];
    auto c = new float[rows * cols];

    fill_ones(a, rows * cols);
    fill_ones(b, rows * cols);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), byte_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), byte_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), byte_size));
    
    cudaMemcpy(d_a, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, byte_size, cudaMemcpyHostToDevice);

    matadd_cublas(d_c, d_a, d_b, rows, cols);
    
    cudaMemcpy(c, d_c, byte_size, cudaMemcpyDeviceToHost);

    print_matrix(rows, cols, c, cols);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}

void test_cublas2() {
    // testing in-place geam
    cublasCreate(&cublas_handle);

    int rows = 3;
    int cols = 2;
    int byte_size = rows * cols * sizeof(float);

    auto a = new float[rows * cols];
    auto b = new float[rows * cols];
    auto c = new float[rows * cols];

    fill_increment<float>(a, rows * cols);
    fill_increment<float>(b, rows * cols);
    fill_zeros<float>(c, rows * cols);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), byte_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), byte_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), byte_size));
    
    cudaMemcpy(d_a, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, byte_size, cudaMemcpyHostToDevice);

    matadd_cublas(d_a, d_a, d_b, rows, cols);
    // CUBLAS_CHECK(cublasSgeam(
    //     cublas_handle,
    //     CUBLAS_OP_N, CUBLAS_OP_N,
    //     rows, cols, &ALPHA, d_a, rows,
    //     &GEAM_BETA, d_b, rows,
    //     d_c, rows
    // ));
    
    cudaMemcpy(c, d_a, byte_size, cudaMemcpyDeviceToHost);

    print_matrix(rows, cols, c, cols);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char** argv) {
    // test1();
    // test2();
    // test_cublas1();
    test_cublas2();
}
