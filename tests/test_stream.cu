#include <string>
#include <cuda_runtime.h>

const int SIZE = 10000;
const int STREAMS = 3;

void test0() {
    float* host_ptrs_array[STREAMS] = {new float[SIZE], new float[SIZE], new float[SIZE]};
    float* dev_ptrs_arr[STREAMS];

    for (int i = 0; i < STREAMS; ++i) {
        cudaMalloc(&dev_ptrs_arr[i], sizeof(float) * SIZE);
        cudaMemcpy(dev_ptrs_arr[i], host_ptrs_array[i], sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < STREAMS; ++i) {
        cudaFree(dev_ptrs_arr[i]);
        delete[] host_ptrs_array[i];
    }
}

void test1() {
    cudaStream_t stream_array[3];

    float* host_ptrs_arr[STREAMS];
    float* dev_ptrs_arr[STREAMS];

    for (int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&stream_array[i]);
    }
    for (int i = 0; i < STREAMS; ++i) {
        cudaMallocAsync(&dev_ptrs_arr[i], sizeof(float) * SIZE, stream_array[i]);
        cudaMemcpyAsync(dev_ptrs_arr[i], host_ptrs_arr[i], sizeof(float) * SIZE, cudaMemcpyHostToDevice, stream_array[i]);
    }
    for (int i = 0; i < STREAMS; ++i) {
        cudaFreeAsync(dev_ptrs_arr[i], stream_array[i]);
        cudaFreeHost(host_ptrs_arr[i]);
        cudaStreamDestroy(stream_array[i]);
    }
}

int main(int argc, char** argv) {
    if (std::stoi(argv[1]) == 0) {
        test0();
    }
    else if (std::stoi(argv[1]) == 1) {
        test1();
    }
    printf("%s\n", argv[1]);
}
