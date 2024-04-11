#include <iostream>
#include "../src/utils.h"

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    int N = 1;
    auto A = new float;
    auto B = new float;
    auto C = new float;
    VecAdd<<<1, N>>>(A, B, C);

    for (int i = 0; i < N; ++i) {
        std::cout << C[i] << std::endl;
    }

    std::cout << "end" << std::endl;
    
    return 0;
}
