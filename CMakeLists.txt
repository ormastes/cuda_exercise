cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(demo)
add_subdirectory(demo_lib)

