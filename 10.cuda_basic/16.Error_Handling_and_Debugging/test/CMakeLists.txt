

add_executable(${MODULE}_test
    unit/kernels/test_vector_add_2d.cu
)

# Link test with testing frameworks
target_link_libraries(${MODULE}_test
    PRIVATE
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
    ${MODULE}_lib_INTERFACE
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_test)

# Add profiling targets
add_profiling_targets(${MODULE}_test)

# Part-specific test executable for error handling demonstrations
add_executable(${MODULE}_part_specific_test
    unit/part_specific/test_error_handling.cu
)

target_link_libraries(${MODULE}_part_specific_test
    PRIVATE
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
    ${MODULE}_lib_INTERFACE
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_part_specific_test)

# Add profiling targets
add_profiling_targets(${MODULE}_part_specific_test)