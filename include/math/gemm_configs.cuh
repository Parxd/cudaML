#include "cutlass/gemm/device/gemm.h"

using MMAOp = cutlass::arch::OpClassSimt;
using SMArch = cutlass::arch::Sm86;
using GemmHalf = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    MMAOp, SMArch
>;
using GemmFloat = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    MMAOp, SMArch
>;
using GemmDouble = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double,
    MMAOp, SMArch
>;
using GemmTF32 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t,
    MMAOp, SMArch
>;
