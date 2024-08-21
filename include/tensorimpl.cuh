#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include "memory.cuh"

using namespace cute;

template <typename T>
class TensorImpl {
    public:
        TensorImpl() = default;
        TensorImpl(int, int);
        TensorImpl(const TensorImpl&);  // use cute::copy algo
        ~TensorImpl() = default;
        TensorImpl operator=(const TensorImpl&);

        void print();
        void print_tensor(cudaStream_t stream);
    private:
        std::shared_ptr<DeviceAlloc<T>> m_alloc;
        // lot of compile-time enforcements; fine for now
        Layout<Shape<int, int>, tuple<int, int>> m_layout;
        Tensor<ViewEngine<T*>, Layout<Shape<int, int>, tuple<int, int>>> m_view;
};

/*
row-major constructor (for now)
*/
template <typename T>
TensorImpl<T>::TensorImpl(int _rows, int _cols) {
    m_alloc = std::make_shared<DeviceAlloc<T>>(_rows * _cols);
    // layout composed of shape + stride
    m_layout = Layout<Shape<int, int>, tuple<int, int>>(
        Shape<int, int>(_rows, _cols),
        Stride<int, int>{_cols, Int<1>{}}
    );
    // tensor composed of engine + layout
    m_view = Tensor<
        ViewEngine<T*>,
        Layout<Shape<int, int>, tuple<int, int>>
    >(
        m_alloc.get()->get(),
        m_layout
    );
}

template <typename T>
TensorImpl<T>::TensorImpl(const TensorImpl& other) {
    
}

/*
view tensor metadata
*/
template <typename T>
void TensorImpl<T>::print() {
    cute::print(m_view);
}

/*
naive implementation for now (very slow)
view tensor contents
*/
template <typename T>
void TensorImpl<T>::print_tensor(cudaStream_t stream) {
    T tmp[m_alloc.get()->size()];
    m_alloc.get()->cpy_from_buffer(tmp, stream);
    auto tmp_view = cute::make_tensor((T*)tmp, m_layout);
    cute::print_tensor(tmp_view);
}
