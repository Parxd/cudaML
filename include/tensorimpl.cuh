#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <memory>
#include "memory.cuh"

using namespace cute;

template <typename T>
class TensorImpl {
    public:
        TensorImpl() = default;
        TensorImpl(int, int);
        TensorImpl(const TensorImpl&);  // use cute::copy algo
        ~TensorImpl();
        TensorImpl operator=(const TensorImpl&);
    private:
        DeviceAlloc<T> m_alloc;
        // lot of compile-time enforcements; fine for now
        Layout<Shape<int, int>, tuple<int, _1>> m_layout;
        Tensor<ViewEngine<T*>, Layout<Shape<int, int>, tuple<int, int>>> m_view;
};

template <typename T>
TensorImpl<T>::TensorImpl(int _rows, int _cols) {
    using lt = Layout<Shape<int, int>, tuple<int, int>>;
    auto linear_size = _rows * _cols;

    m_alloc = DeviceAlloc<T>(linear_size);
    // m_layout = make_layout(make_shape(_rows, _cols), LayoutRight{});
    // m_view = Tensor<ViewEngine<T*>, lt>(m_alloc.get(), m_layout);
}

template <typename T>
TensorImpl<T>::TensorImpl(const TensorImpl& other) {
    
}

template <typename T>
TensorImpl<T>::~TensorImpl() {

}
