cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA toolkit root
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CUDAToolkit_ROOT /usr/local/cuda-13.0)

# Set CUDA architectures before project
set(CMAKE_CUDA_ARCHITECTURES 75;80;86)

# Enable CUDA language
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Testing support
enable_testing()

# FetchContent for downloading Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include Google Test's CMake utilities
include(CTest)
include(GoogleTest)

# Global settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    # Profile flags: Optimized with debug info
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O2>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O2>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release flags: Optimize for performance
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O3;-use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O3>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
    option(ENABLE_PROFILING "Enable profiling instrumentation" OFF)
endif()

# Include profiling utilities
include(${CMAKE_CURRENT_SOURCE_DIR}/00.cmake_lib/inspect_profile.cmake)

# Verbose PTX compilation (shows register usage)
option(VERBOSE_PTXAS "Show PTX compilation details" OFF)
if(VERBOSE_PTXAS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()


add_subdirectory(00.cuda_custom_lib)
add_subdirectory(00.demo)
add_subdirectory(00.demo_lib)
add_subdirectory(00.test_lib)
add_subdirectory(10.cuda_basic)
add_subdirectory(20.cuda_intermediate)
