#include <cublas_v2.h>
#include "math/add.cu"
#include "math/mat_mul.cu"

bool HOST_PINNED_MEMORY = true;

template <typename T>
struct mat {
    T* data = nullptr;
    T* grad = nullptr;
    T* d_data = nullptr;
    T* d_grad = nullptr;
    void free() {
        if (HOST_PINNED_MEMORY) {
            if (data) cudaFreeHost(data);
            if (grad) cudaFreeHost(grad);
        }
        else {
            if (data) delete[] data;
            if (grad) delete[] grad;
        }
        if (d_data) cudaFree(d_data);
        if (d_grad) cudaFree(d_grad);
    }
};

int main(int argc, char** argv) {
    cublasCreate(&cublas_handle);
    const int config[4] = {3, 5, 5, 2};
    const int batch_size = 32;

    auto w_1 = mat<double>();
    auto b_1 = mat<double>();
    auto w_2 = mat<double>();
    auto b_2 = mat<double>();

    cudaMallocHost((void**)&w_1.data, sizeof(float) * config[1] * config[0]);
    cudaMallocHost((void**)&w_1.grad, sizeof(float) * config[1] * config[0]);
    cudaMallocHost((void**)&b_1.data, sizeof(float) * );
    cudaMallocHost((void**)&b_1.data, sizeof(float) * );
    cudaMallocHost((void**)&w_2.data, sizeof(float) * config[2] * config[1]);
    cudaMallocHost((void**)&w_2.grad, sizeof(float) * config[2] * config[1]);
    cudaMallocHost((void**)&b_2.data, sizeof(float) * );
    cudaMallocHost((void**)&b_2.data, sizeof(float) * );

    w_1.free();
    b_1.free();
    w_2.free();
    b_2.free();

    return 0;
}
