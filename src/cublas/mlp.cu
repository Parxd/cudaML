#include <assert.h>

#include <cublas_v2.h>
#include "add.cu"
#include "mat_mul.cu"
#include "linear.cu"
#include "../utils.h"
#include "../../include/cmdparser.hpp"

using dtype = float;
const int batch_size = 32;

template <typename T>
__global__ void relu_forward_kernel(T* input, T* output, int M, int N) {
    // each warp handles one row of the input
    int warpsPerBlock = blockDim.x / warpSize;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int numWarps = gridDim.x * warpsPerBlock;
    for (int row = blockIdx.x * warpsPerBlock + warpId; row < M;
         row += numWarps)
        if (row < M) {
            T* const x = input + row * N;
            T* const y = output + row * N;

            for (int i = laneId; i < N; i += warpSize) {
                y[i] = x[i] > 0 ? x[i] : 0.0f;
            }
        }
}

template <typename T>
struct Layer_t {
    // TODO: allocate contiguous mem chunks (?)
    T* weight = nullptr;
    T* bias = nullptr;
    T* z = nullptr;
    T* fwd = nullptr;
    int in_features, out_features;
    size_t bytes;
    size_t fwd_size;

    Layer_t(const int& in, const int& out): in_features(in), out_features(out), bytes(sizeof(T) * in * out), fwd_size(sizeof(T) * batch_size * out) {}
    ~Layer_t() = default;  // TODO: will do manual mem management for now b/c need to handle differently for host vs. device
};

template <typename T>
void allocateHostLayer(Layer_t<T>& layer) {
    CUDA_CHECK(cudaMallocHost((void**)&layer.weight, layer.bytes)); 
    CUDA_CHECK(cudaMallocHost((void**)&layer.bias, layer.fwd_size));
    CUDA_CHECK(cudaMallocHost((void**)&layer.z, layer.fwd_size));
    CUDA_CHECK(cudaMallocHost((void**)&layer.fwd, layer.fwd_size)); 
}

template <typename T>
void allocateDeviceLayer(Layer_t<T>& layer, cudaStream_t stream) {
    CUDA_CHECK(cudaMallocAsync((void**)&layer.weight, layer.bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&layer.bias, layer.fwd_size, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&layer.z, layer.fwd_size, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&layer.fwd, layer.fwd_size, stream));
}

template <typename T>
void fillLayerRandom(Layer_t<T>& layer) {
    // fill_random_uniform<T>(layer.weight, layer.in_features * layer.out_features);
    fill_ones<T>(layer.weight, layer.in_features * layer.out_features);
    // fill_random_uniform<T>(layer.bias, layer.out_features);  // can't use until we have broadcasting addition
    
    fill_ones<T>(layer.bias, layer.out_features);
    // fill_random_uniform<T>(layer.bias, layer.out_features);
    int j = 0;
    for (int i = layer.out_features; i < batch_size * layer.out_features; ++i) {
        if (j == layer.out_features) {
            j = 0;
        }
        layer.bias[i] = layer.bias[j];
        ++j;
    }

    // print_matrix(layer.out_features, layer.in_features, layer.weight, layer.in_features);
    // std::cout << "\n";
    // print_matrix(batch_size, layer.out_features, layer.bias, layer.out_features);
}

template <typename T>
void freeHostLayer(Layer_t<T>& layer) {
    CUDA_CHECK(cudaFreeHost(layer.weight));
    CUDA_CHECK(cudaFreeHost(layer.bias));
    CUDA_CHECK(cudaFreeHost(layer.z));
    CUDA_CHECK(cudaFreeHost(layer.fwd));
}

template <typename T>
void freeDeviceLayer(Layer_t<T>& layer, cudaStream_t stream) {
    CUDA_CHECK(cudaFreeAsync(layer.weight, stream));
    CUDA_CHECK(cudaFreeAsync(layer.bias, stream));
    CUDA_CHECK(cudaFreeAsync(layer.z, stream));
    CUDA_CHECK(cudaFreeAsync(layer.fwd, stream));
}

template <typename T>
void copyDeviceLayer(Layer_t<T>& host_layer, Layer_t<T>& dev_layer, cudaStream_t stream) {
    // host -> dev
    CUDA_CHECK(cudaMemcpyAsync(dev_layer.weight, host_layer.weight, host_layer.bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_layer.bias, host_layer.bias, host_layer.fwd_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_layer.z, host_layer.z, host_layer.fwd_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dev_layer.fwd, host_layer.fwd, host_layer.fwd_size, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void copyHostLayer(Layer_t<T>& dev_layer, Layer_t<T>& host_layer, cudaStream_t stream) {
    // dev -> host
    CUDA_CHECK(cudaMemcpyAsync(host_layer.weight, dev_layer.weight, host_layer.bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_layer.bias, dev_layer.bias, host_layer.fwd_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_layer.z, dev_layer.z, host_layer.fwd_size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_layer.fwd, dev_layer.fwd, host_layer.fwd_size, cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void forward(Layer_t<T> layer, T* X, cudaStream_t stream) {
    // linear_forward(layer.z, layer.weight, X, layer.bias, batch_size, layer.in_features, layer.out_features);
    matmul_cublas(layer.z, X, layer.weight, false, true, batch_size, layer.in_features, layer.out_features);
    // add_cublask(layer.z, layer.z, layer.bias, batch_size, layer.out_features);
    // relu_forward_kernel<<<batch_size * layer.out_features / 128, 128>>>(layer.z, layer.fwd, batch_size, layer.out_features);
}

int main(int argc, char** argv) {
    cli::Parser parser(argc, argv);
    // TODO: parse cli args

    cublasCreate(&cublas_handle);

    // setup stream pool
    const int STREAMS = 4;
    const int config[4] = {2, 5, 10, 2};
    cudaStream_t stream_arr[STREAMS];
    for (int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&stream_arr[i]);
    }
    cublasSetStream(cublas_handle, stream_arr[0]);

    Layer_t<dtype> layer1(config[0], config[1]);
    Layer_t<dtype> d_layer1(config[0], config[1]);

    allocateHostLayer(layer1);
    fillLayerRandom(layer1);
    allocateDeviceLayer(d_layer1, stream_arr[0]);
    copyDeviceLayer(layer1, d_layer1, stream_arr[0]);

    dtype* X;
    dtype* Y;
    size_t X_bytes = sizeof(dtype) * batch_size * layer1.in_features;
    size_t Y_bytes = sizeof(dtype) * layer1.out_features * layer1.in_features;
    // X: 32 x 2
    // Y: 5 x 2
    // Y^T: 2 x 5

    CUDA_CHECK(cudaMallocHost((void**)&X, X_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&Y, Y_bytes));
    fill_ones<dtype>(X, batch_size * layer1.in_features);
    fill_ones<dtype>(Y, layer1.out_features * layer1.in_features);

    dtype* d_X;
    dtype* d_Y;
    dtype* d_Z;
    int Z_bytes = sizeof(dtype) * batch_size * layer1.out_features;
    CUDA_CHECK(cudaMallocAsync((void**)&d_X, X_bytes, stream_arr[0]));
    CUDA_CHECK(cudaMallocAsync((void**)&d_Y, Y_bytes, stream_arr[0]));
    CUDA_CHECK(cudaMallocAsync((void**)&d_Z, Z_bytes, stream_arr[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_X, X, X_bytes, cudaMemcpyHostToDevice, stream_arr[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_Y, Y, Y_bytes, cudaMemcpyHostToDevice, stream_arr[0]));
    matmul_cublas(d_Z, d_X, d_Y, false, true, batch_size, layer1.in_features, layer1.out_features);

    cudaDeviceSynchronize();
    
    auto Z = new dtype[batch_size * layer1.out_features];
    CUDA_CHECK(cudaMemcpy(Z, d_Z, Z_bytes, cudaMemcpyDeviceToHost));
    // copyHostLayer(d_layer1, layer1, stream_arr[0]);
    // print_matrix<dtype>(batch_size, layer1.out_features, Z, layer1.out_features);

    cudaFreeHost(X);
    cudaFree(d_X);
    freeHostLayer(layer1);
    freeDeviceLayer(d_layer1, stream_arr[0]);

    // Layer_t<dtype> layer2(config[0], config[1]);
    // Layer_t<dtype> d_layer2(config[0], config[1]);
    
    // allocateHostLayer(layer1);
    // allocateHostLayer(layer2);
    // fillLayerRandom(layer1);
    // fillLayerRandom(layer2);

    // allocateDeviceLayer(d_layer1, stream_arr[0]);
    // allocateDeviceLayer(d_layer2, stream_arr[1]);    

    // copyDeviceLayer(layer1, d_layer2, stream_arr[0]);
    // // // TODO: d_layer1 forward pass here on stream_arr[0]
    // copyDeviceLayer(layer2, d_layer2, stream_arr[1]);  // won't run parallel to copyDeviceLayer(layer1) b/c is memcpy blocking

    // freeHostLayer(layer1);
    // freeHostLayer(layer2);
    // freeDeviceLayer(d_layer1, stream_arr[0]);
    // freeDeviceLayer(d_layer1, stream_arr[0]);

    for (int i = 0; i < STREAMS; ++i) {
        cudaStreamDestroy(stream_arr[i]);
    }
    cublasDestroy(cublas_handle);

    return 0;
}
