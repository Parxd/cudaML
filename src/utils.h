#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>

#define MAX_THREADS 32
#define CEIL_DIV(N, M) (((N) + (M)-1) / (M))
const float ALPHA = 1.0;
const float BETA = 0.0;

void fill_ones(float* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = 1.0;
    }
}

template <typename T>
void fill_increment(T* arr, int N) {
    T num = 1.0;
    for (int i = 0; i < N; ++i) {
        arr[i] = num;
        ++num;
    }
}

// STRIDE--number of elements to skip to get to next row (number of columns in row-major)
template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &stride);
template <> void print_matrix(const int &m, const int &n, const float *A, const int &stride) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.3f\t", A[i * stride + j]);
        }
        std::printf("\n");
    }
}
template <> void print_matrix(const int &m, const int &n, const double *A, const int &stride) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.3f\t", A[i * stride + j]);
        }
        std::printf("\n");
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

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
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

#endif  // UTILS_H
