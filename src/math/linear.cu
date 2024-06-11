#include "add.cu"
#include "mat_mul.cu"

void linear_forward(float* out, float* w, float* x, float* b,
                    int row, int inner, int col) {
    // out = x @ W^T + b
    matmul_cublas(out, x, w, false, true, row, inner, col);
    add_cublas(out, out, b, row, col);
}
