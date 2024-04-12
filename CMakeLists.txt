cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(cuda LANGUAGES CXX CUDA)
include(CheckLanguage)
enable_language(CUDA)

find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(MAIN_SOURCES
    src/math/vec_add.cu
    src/math/mat_add.cu
    )

add_executable(test_vec_add tests/math/test_vec_add.cu)
add_executable(test_mat_add tests/math/test_mat_add.cu)
