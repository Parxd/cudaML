#include <cublas_v2.h>
#include "math/add.cu"
#include "math/mat_mul.cu"

int main(int argc, char** argv) {
    cublasCreate(&cublas_handle);
    int config[5] = {3, 5, 10, 5, 2};
    int batch_size = 32;
    
    return 0;
}
