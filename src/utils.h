#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void fill_ones(float* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = 1.0;
    }
}

void print_matrix(float* arr, int rows, int cols, int precision = 3) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.3f ", *(arr + i * rows + j));
        }
        printf("\n");
    }
}

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define CUDNN_CHECK(err)                                                                           \
    do {                                                                                           \
        cudnnStatus_t err_ = (err);                                                                \
        if (err_ != CUDNN_STATUS_SUCCESS) {                                                        \
            std::printf("cuDNN error %d at %s:%d\n", err_, __FILE__, __LINE__);                    \
            throw std::runtime_error("cuDNN error");                                               \
        }                                                                                          \
    } while (0)
