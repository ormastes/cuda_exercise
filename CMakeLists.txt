cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES native) # 70, 86, native, all-major

# Workaround for NVCC + Clang detection issues
if(DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_LINK_FLAGS "-L/usr/local/cuda/lib64 -lcudart")
endif()

project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Testing support
enable_testing()

# FetchContent for downloading Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# Include Catch2's CMake utilities
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# Global settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
    )
endif()

add_subdirectory(00.demo)
add_subdirectory(00.demo_lib)
