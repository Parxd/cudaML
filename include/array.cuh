enum device {
    cpu = 0,
    gpu = 1
};

struct Array
{
    size_t m_rows;
    size_t m_cols;
    size_t m_byte_size;
    float* m_host = nullptr;
    float* m_device = nullptr;
    device m_current;

    Array(float*, int, int, bool);
    ~Array();
    void print(int, std::ostream&);
    Array operator+(const Array&) const;

    void to_host();
    void to_device();
};
