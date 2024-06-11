#include <cublas_v2.h>
#include "../../src/utils.h"

void test0() {
    cublasCreate(&cublas_handle);
    int size = 10;
    float* a = new float[size];
    fill_increment<float>(a, size);

    float* d_a;
    cudaMalloc((void**)&d_a, sizeof(float) * size);
    cudaMemcpy(d_a, a, sizeof(float) * size, cudaMemcpyHostToDevice);

    float res;
    cublasSasum_v2(cublas_handle, size, d_a, 1, &res);
   
    cublasDestroy(cublas_handle);
    delete[] a;
    cudaFree(d_a);

    std::cout << res << std::endl;
}

int main(int argc, char** argv) {
    test0();
}