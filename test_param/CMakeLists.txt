cmake_minimum_required(VERSION 3.18)
project(test_gpu_parameterized CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Fetch GTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Add test executables
add_executable(test_gpu_parameterized test_gpu_parameterized.cu)
add_executable(test_gpu_generator_simple test_gpu_generator_simple.cu)

# Set CUDA architectures (detect or use a common one)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
endif()
set_target_properties(test_gpu_parameterized PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

set_target_properties(test_gpu_generator_simple PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Link libraries
target_link_libraries(test_gpu_parameterized
    gtest
    gtest_main
)

target_link_libraries(test_gpu_generator_simple
    gtest
    gtest_main
)

# Include directories
target_include_directories(test_gpu_parameterized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(test_gpu_generator_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Add as test
enable_testing()
add_test(NAME GpuParameterizedTests COMMAND test_gpu_parameterized)
add_test(NAME GpuGeneratorTests COMMAND test_gpu_generator_simple)