#include <assert.h>
#include "utils.h"
#include "../include/array.cuh"

array::array(float* data, int N, int rows, int cols) {
    assert(N == rows * cols);
    assert(data != nullptr);
    m_buffer = new float[N];  // should probably catch for new failure
    for (int i = 0; i < N; ++i) {
        m_buffer[i] = data[i];
    }
    m_rows = rows;
    m_cols = cols;
    m_size = N;
    m_byte_size = sizeof(float) * m_size;
    m_gpu = false;
}
array::array(std::vector<float>& data, int rows, int cols) {
    array(data.data(), data.size(), rows, cols);
}
array::~array() noexcept {
    delete[] m_buffer;
}
void array::print(int precision = 3) const noexcept {
    print_matrix(m_rows, m_cols, m_buffer, m_cols, precision);
}
