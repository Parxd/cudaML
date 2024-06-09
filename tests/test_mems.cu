#include <assert.h>
#include <vector>
#include "../include/memory.cuh"

void test1() {
    auto data = new float[2 * 3];
    host_mem<float> A(data, 2, 3);
}

void test2() {
    
}

int main(int argc, char** argv) {    
    test1();
    // test2();
    return 0;
}
