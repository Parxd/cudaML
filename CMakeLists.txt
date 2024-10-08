cmake_minimum_required(VERSION 3.20)

project(cudaML LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/cutlass/tools/util/include")
    message(FATAL_ERROR [[
        CUTLASS library not found in third-partye/cutlass.
        Please fetch with git submodule update --init --recursive
        ]])
endif()

# CUDA
include(CheckLanguage)
check_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 86)  # only support architectures present in system, for now


include_directories(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/util/include
                    ${PROJECT_SOURCE_DIR}/third-party/cutlass/include
                    ${PROJECT_SOURCE_DIR}/third-party/cutlass/tools/util/include
)

# Compiler flags
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND LLM_CUDA_NVCC_FLAGS -lineinfo)
    list(APPEND LLM_CUDA_CLANG_FLAGS -gmlt)
endif()
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")                         # verbose
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=-fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")   # CuTe layout.hpp warning

message(STATUS "build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Using NVCC flags: ${GEMM_CUDA_NVCC_FLAGS}")
message(STATUS "cxx compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "cxx compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

# cuBLAS (optional)
option(ENABLE_CUBLAS "Enable cuBLAS support" OFF)
if(TARGET CUDA::cublas)
    message(STATUS "cuBLAS found.")
    set(ENABLE_CUBLAS ON)
else()
    message(STATUS "cuBLAS not found, some code and tests will not be compiled.")
endif()

add_subdirectory(src)
add_subdirectory(tests)