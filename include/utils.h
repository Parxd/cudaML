#ifndef UTILS
#define UTILS

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cublas_v2.h>

#define MAX_THREADS 32
#define CEIL_DIV(N, M) (((N) + (M)-1) / (M))
static cublasHandle_t cublas_handle;
const float ALPHA = 1.0;
const float BETA = 0.0;
const float GEAM_BETA = 1.0;

template <typename T>
void fill_zeros(T* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = T(0);
    }
}

template <typename T>
void fill_ones(T* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = T(1);
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

template <typename T>
bool is_device_pointer(T* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    if (error != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return (attributes.type == cudaMemoryTypeDevice);
}

template <typename T>
void fill_random_uniform(T* arr, int N, double min=-1, double max=1) {
    std::mt19937_64 rng;
    std::uniform_real_distribution<T> distribution(min, max);
    uint64_t time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(time_seed & 0xffffffff), uint32_t(time_seed>>32)};
    rng.seed(ss);

    for (int i = 0; i < N; ++i) {
        arr[i] = distribution(rng);
    }
}

// STRIDE--number of elements to skip to get to next row (number of columns in row-major)
template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &stride, const int& precision = 3);
template <> void print_matrix(const int &m, const int &n, const float *A, const int &stride, const int& precision) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.*f\t", precision, A[i * stride + j]);
        }
        std::printf("\n");
    }
}
template <> void print_matrix(const int &m, const int &n, const double *A, const int &stride, const int& precision) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.*f\t", precision, A[i * stride + j]);
        }
        std::printf("\n");
    }
}

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == nullptr) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

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

#endif