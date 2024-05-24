#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/array.cuh"
#include "utils.h"
#include "math/mat_add.cu"
#include "math/mat_mul.cu"
#include "math/linear.cu"

Array::Array(float* data, int rows, int cols, bool host_array = true) {
    m_rows = rows;
    m_cols = cols;
    if (host_array) {
        m_host = data;
        m_current = cpu;
    }
    else {
        m_device = data;
        m_current = gpu;
    }
    m_byte_size = sizeof(float) * rows * cols;
}

Array::~Array() {
    if (m_host) {delete[] m_host;}
    if (m_device) {CUDA_CHECK(cudaFree(m_device));}
}

Array Array::operator+(const Array& other) const {
    float* res_device;
    CUDA_CHECK(cudaMalloc((void**)&res_device, m_byte_size));
    matadd_cublas(res_device, m_device, other.m_device, m_rows, m_cols);
    return Array(res_device, m_rows, m_cols, false);
}

void Array::print(int precision = 3, std::ostream &os = std::cout) {
    if (!m_host) {
            m_host = new float[m_rows * m_cols];
        }
    CUDA_CHECK(cudaMemcpy(m_host, m_device, m_byte_size, cudaMemcpyDeviceToHost));
    print_matrix(m_rows, m_cols, m_host, m_cols, precision);
}

void Array::to_host() {
    if (m_current == gpu) {
        // first-time alloc (device -> host)
        if (!m_host) {
            m_host = new float[m_rows * m_cols];
        }
        CUDA_CHECK(cudaMemcpy(m_host, m_device, m_byte_size, cudaMemcpyDeviceToHost));
        m_current = cpu;
    }
}

void Array::to_device() {
    if (m_current == cpu) {
        // first-time alloc (host -> device)
        if (!m_device) {
            CUDA_CHECK(cudaMalloc((void**)&m_device, m_byte_size));
        }
        CUDA_CHECK(cudaMemcpy(m_device, m_host, m_byte_size, cudaMemcpyHostToDevice));
        m_current = gpu;
    }
}
