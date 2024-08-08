#include "../../include/utils.h"
#include "../../src/cublas/linear.cu"

void test_linear1() {
    cublasCreate(&cublas_handle);
    float* weight = (float*)mallocCheck(sizeof(float) * 5 * 3);
    float* bias = (float*)mallocCheck(sizeof(float) * 5);
    float* input = (float*)mallocCheck(sizeof(float) * 3);
    float* fwd = (float*)mallocCheck(sizeof(float) * 5);

    fill_increment<float>(weight, 15);
    fill_ones<float>(bias, 5);
    fill_increment<float>(input, 3);
    fill_zeros<float>(fwd, 5);

    float* d_weight, *d_bias, *d_input, *d_fwd;
    CUDA_CHECK(cudaMalloc((void**)&d_weight, sizeof(float) * 5 * 3));
    CUDA_CHECK(cudaMalloc((void**)&d_bias, sizeof(float) * 5));
    CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(float)* 3));
    CUDA_CHECK(cudaMalloc((void**)&d_fwd, sizeof(float) * 5));

    CUDA_CHECK(cudaMemcpy(d_weight, weight, sizeof(float) * 5 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, sizeof(float) * 5, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, sizeof(float) * 3, cudaMemcpyHostToDevice));

    linear_forward(d_fwd, d_weight, d_input, d_bias, 1, 3, 5);

    CUDA_CHECK(cudaMemcpy(fwd, d_fwd, sizeof(float) * 5, cudaMemcpyDeviceToHost));

    print_matrix(1, 5, fwd, 5);
    // expected: [14, 32, 50, 68, 86] + [1, 1, 1, 1, 1] = [15, 33, 51, 69, 87]

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_fwd));
    free(weight);
    free(bias);
    free(input);
    free(fwd);
    cublasDestroy(cublas_handle);
}

int main(int argc, char** argv)
{
    test_linear1();
}
