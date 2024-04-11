#include <iostream>
#include <cudnn.h>


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


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
