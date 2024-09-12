cmake_minimum_required(VERSION 3.20)

add_executable(2.x_gemm cutlass_2.x_gemm.cu)
add_executable(test_driver main.cu)