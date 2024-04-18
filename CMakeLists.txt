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
    src/math/mat_mul.cu
    )

set(TEST_SOURCES
    tests/math/test_vec_add.cu
    tests/math/test_mat_add.cu
    tests/math/test_mat_mul.cu
    )

add_executable(main ${MAIN_SOURCES})

foreach(TEST_SOURCE ${TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_link_libraries(${TEST_NAME} cublas)
endforeach()
