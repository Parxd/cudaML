#include <vector>

struct array
{
    size_t m_rows;
    size_t m_cols;
    size_t m_size;
    size_t m_byte_size;
    float* m_buffer;
    bool m_gpu;
    array() {}
    array(float* data, int N, int rows, int cols);
    array(std::vector<float>& data, int rows, int cols);
    array(const array&);
    ~array() noexcept;
    void print(int precision) const noexcept;
};
