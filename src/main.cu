#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/axpby.hpp>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include "../include/tensorimpl.cuh"

using namespace cutlass;

template <typename TensorType>
bool is_device_tensor(const TensorType& tensor) {
    return is_device_pointer(tensor.data());
}

int main(int argc, char* argv[]) {
    using dtype = float;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // int M = 10;
    // int N = 10;
    // int K = 10;
    // auto a_ptr = new float[M * K];
    // auto b_ptr = new float[K * N];
    // auto c_ptr = new float[M * N];
    
    // for (int i = 0; i < M * K; ++i) {
    //     a_ptr[i] = 1;
    //     b_ptr[i] = 2;
    // }

    // auto a = cute::make_tensor(a_ptr, cute::make_shape(M, K), cute::make_stride(M, 1));
    // auto b = cute::make_tensor(b_ptr, cute::make_shape(K, N), cute::make_stride(K, 1));
    // auto c = cute::make_tensor(c_ptr, cute::make_shape(M, N), cute::make_stride(M, 1));
    // // cute::axpby(1, a, 1, b);
    // cute::print_tensor(b);

    auto tensor = TensorImpl<dtype>(2, 3);  // 2 x 3
    tensor.print_tensor(stream);

    auto size = 50;
    auto alloc_ptr = std::make_shared<DeviceAlloc<dtype>>(size);
    dtype src[size];
    for (int i = 0; i < size; ++i) {
        src[i] = (dtype)1.5;
    }
    alloc_ptr.get()->cpy_to_buffer(src, size, stream);
    dtype tmp[alloc_ptr.get()->size()];
    alloc_ptr.get()->cpy_from_buffer(tmp, stream);
    // for (int i = 0; i < size; ++i) {
        // std::cout << tmp[i] << " ";
    // }
    auto layout = Layout<Shape<int, int>, tuple<int, int>>(
        Shape<int, int>(5, 10),
        Stride<int, int>{10, Int<1>{}}
    );
    auto test = Tensor<ViewEngine<dtype*>, Layout<Shape<int, int>, tuple<int, int>>>((float*)tmp, layout);
    cute::print_tensor(test);
    // delete[] a_ptr;
    // delete[] b_ptr;
    // delete[] c_ptr;
    cudaStreamDestroy(stream);
    std::cout << std::endl;
}
