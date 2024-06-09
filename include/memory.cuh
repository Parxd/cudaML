#ifndef MEMORY
#define MEMORY

template <typename dtype>
struct host_mem
{
    size_t _rows;
    size_t _cols;
    size_t _bytes;
    dtype* _raw_buffer;
    host_mem(dtype*, int, int);
    host_mem(std::vector<dtype>, int, int);
};
template <typename dtype>
struct device_mem
{
    size_t _rows;
    size_t _cols;
    size_t _bytes;
    dtype* _raw_buffer;
};

template <typename dtype>
host_mem<dtype>::host_mem(dtype* data, int rows, int cols) {
    _rows = rows;
    _cols = cols;
    _bytes = sizeof(dtype) * _rows * _cols;
}
template <typename dtype>
host_mem<dtype>::host_mem(std::vector<dtype> data, int rows, int cols) {
    _rows = rows;
    _cols = cols;
    _bytes = sizeof(dtype) * _rows * _cols;
}

#endif