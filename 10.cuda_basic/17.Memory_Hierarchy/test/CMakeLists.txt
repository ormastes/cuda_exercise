# Test executable for memory hierarchy demonstrations
add_executable(${MODULE}_test
    kernels/test_matrix_multiply.cu
    kernels/test_vector_add_2d.cu
)

# Add include directories for test headers
target_include_directories(${MODULE}_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/00.test_lib  # For gpu_gtest.h
)

# Link test with testing frameworks
target_link_libraries(${MODULE}_test
    PRIVATE
    ${MODULE}_lib
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_test)

# Add profiling targets
add_profiling_targets(${MODULE}_test)

# Performance benchmark executable
add_executable(${MODULE}_benchmark
    part_specific/benchmark_memory.cu
    part_specific/test_vector_add_memory.cu
)

target_include_directories(${MODULE}_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/00.test_lib  # For gpu_gtest.h
)

target_link_libraries(${MODULE}_benchmark
    PRIVATE
    ${MODULE}_lib
    ${MODULE}_part_specific_lib
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
)

gtest_discover_tests(${MODULE}_benchmark)

add_profiling_targets(${MODULE}_benchmark)

