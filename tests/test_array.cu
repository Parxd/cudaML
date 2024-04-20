#include "../src/array.cu"

int main(int argc, char** argv) {
    float data[] = {1.5, 2.5, 3.5, 4.5};
    array A(data, 4, 2, 2);
    A.print();
    return 0;
}