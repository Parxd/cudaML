#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "cutlass/gemm/device/gemm.h"
#include <cuda_runtime.h>
#include "../tensorimpl.cuh"
#include "gemm_configs.cuh"

template <typename T>
TensorImpl<T> mm(const TensorImpl<T>&, const TensorImpl<T>&, cudaStream_t);

template <>
TensorImpl<float> mm(const TensorImpl<float>& t1, const TensorImpl<float>& t2, cudaStream_t stream) {
    assert(cute::get<1>(t1.m_layout.shape()) == cute::get<0>(t2.m_layout.shape()));
    int M = cute::get<0>(t1.m_layout.shape());
    int N = cute::get<1>(t2.m_layout.shape());
    int K = cute::get<0>(t2.m_layout.shape());
    auto res = TensorImpl<float>(M, N);
    GemmFloat::Arguments arguments{
        {M, N, K},
        {t1.m_alloc->get(), K},
        {t2.m_alloc->get(), N},
        {res.m_alloc->get(), N},
        {res.m_alloc->get(), N},
        {1.0f, 0.0f}
    };
    GemmFloat op;
    op(arguments, stream);
    return res;
}

#endif