#include <assert.h>
#include "../src/array.cu"

void test1() {
    // sanity checks
    auto data_A = new float[6];
    fill_increment<float>(data_A, 6);
    Array A(data_A, 2, 3);

    assert(A.m_rows == 2);
    assert(A.m_cols == 3);
    assert(A.m_current == cpu);
    assert(A.m_host);
    assert(!A.m_device);

    A.to_device();

    assert(A.m_current == gpu);
    assert(A.m_host);  // host ptr should still exist, albeit now considered "garbage" mem
    assert(A.m_device);  // and of course, cuda malloc should have been called

    A.print();  // TODO: test stream output w/ std::stringstream

    assert(A.m_current == gpu);
}

void test2() {
    auto data_A = new float[6];
    auto data_B = new float[6];
    fill_increment<float>(data_A, 6);
    fill_increment<float>(data_B, 6);
    Array A(data_A, 2, 3);
    Array B(data_B, 2, 3);
    auto C = A + B;

    assert(C.m_rows == 2);
    assert(C.m_cols == 3);
    assert(C.m_current == gpu);
    assert(!C.m_host);
    assert(C.m_device);

    // C.print();
}

int main(int argc, char** argv) {
    cublasCreate(&cublas_handle);
    
    // test1();
    test2();
    return 0;
}
