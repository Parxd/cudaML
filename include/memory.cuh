#ifndef MEMORY_CUH
#define MEMORY_CUH

#include <cutlass/util/device_memory.h>
#include <cutlass/util/exceptions.h>
#include <cuda_runtime.h>
#include "utils.h"

// mv primitives
template <typename T>
void async_copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    size_t bytes = count * cutlass::sizeof_bits<T>::value / 8;
    if (bytes == 0 && count > 0) {
        bytes = 1;
    }
    cudaError_t cuda_error = (cudaMemcpyAsync(dst, src, bytes, kind, stream));
    if (cuda_error != cudaSuccess) {
        throw cutlass::cuda_exception("cudaMemcpyAsync() failed", cuda_error);
    }
}

template <typename T>
class DeviceAlloc {
    public:
        DeviceAlloc(): m_buffer(nullptr), m_count(0) {}
        DeviceAlloc(size_t count): m_buffer(nullptr), m_count(count) {
            m_buffer = cutlass::device_memory::allocate<T>(m_count);
            // CUDA_CHECK(cudaMalloc((void**)&m_buffer, sizeof(T) * m_count));
        }
        ~DeviceAlloc() {
            if (m_buffer) {
                cutlass::device_memory::free<T>(m_buffer);
                // CUDA_CHECK(cudaFree(m_buffer));  // suppress compiler warning abt macro
            }
            // cudaDeviceSynchronize();  // TODO: remove this later
            std::cout << m_count << " elements freed from device memory." << std::endl;
        }
        /**
         * Op. assignment needs to be defined for TensorImpl
         * 
         * Shallow or deep copy? 
         * Should multiple DeviceAlloc instances be allowed to point to the same buffer?
         * Should multiple TensorImpl instances be allowed to point to the same buffer?
         * Want potential to "view" the same buffer differently with two TensorImpl instances
         * --> shallow copy + use (something like) std::shared_ptr approach?
         */
        DeviceAlloc& operator=(const DeviceAlloc& other) {

        }
        auto get() const {
            return m_buffer;
        }
        auto size() const {
            return m_count;
        }
        void cpy_to_buffer(T* src, size_t count, cudaStream_t stream) {
            if (count != m_count) {
                throw std::runtime_error("Copy buffer size must match internal buffer size");
            }
            async_copy(get(), src, count, cudaMemcpyHostToDevice, stream);
        }
        void cpy_from_buffer(T* src, cudaStream_t stream) const {
            async_copy(src, get(), m_count, cudaMemcpyDeviceToHost, stream);
        }
    private:
        T* m_buffer;
        size_t m_count;
};

#endif