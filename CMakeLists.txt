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

set(SOURCES
    src/math/vec_add.cu
    )

add_executable(cuda ${SOURCES})

target_include_directories(cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cuda cudnn)
