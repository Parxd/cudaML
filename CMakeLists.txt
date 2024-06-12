cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
enable_language(CUDA)

find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(MAIN_SOURCES

    src/math/vec_add.cu
    src/math/add.cu
    src/math/mat_mul.cu
    )

set(TEST_SOURCES
    tests/test_mems.cu
    tests/test_stream.cu

    tests/math/test_auto.cu
    tests/math/test_sum.cu
    tests/math/test_vec_add.cu
    tests/math/test_add.cu
    tests/math/test_mat_mul.cu
    tests/math/test_linear.cu
    )

foreach(TEST_SOURCE ${TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
  include_directories(CUTLASS_INCLUDE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_link_libraries(${TEST_NAME} cublas)
endforeach()

add_executable(mlp src/mlp.cu)
target_link_libraries(mlp cublas)

add_executable(cutlass_test src/cutlass_test.cu)
include_directories(../cutlass-3.4.1/include)
target_compile_features(cutlass_test PUBLIC cxx_std_17)
target_link_libraries(cutlass_test cublas)
