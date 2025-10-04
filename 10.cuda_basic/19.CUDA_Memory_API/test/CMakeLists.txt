# Tests for Part 19 - CUDA Memory API

# Test executable
add_executable(${MODULE}_test
    unit/kernels/test_matrix_multiply.cu
    unit/kernels/test_vector_add_2d.cu
    unit/part_specific/test_memory_api.cu
)

target_link_libraries(${MODULE}_test PRIVATE
    ${MODULE}_lib
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
    GTestCudaGenerator
)

# Add tests to CTest
gtest_discover_tests(${MODULE}_test)

# Add profiling targets for tests
add_profiling_targets(${MODULE}_test)
