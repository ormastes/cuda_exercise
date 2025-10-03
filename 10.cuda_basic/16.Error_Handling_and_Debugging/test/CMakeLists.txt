
# Part-specific test executable for error handling demonstrations
add_executable(${MODULE}_test
    part_specific/test_error_handling.cu
)

# Link test with testing frameworks
target_link_libraries(${MODULE}_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib  # For cuda_utils.h utilities
    ${MODULE_NAME}_lib_INTERFACE
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_test)

# Add profiling targets
add_profiling_targets(${MODULE}_test)

# Debug build specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${MODULE}_test PRIVATE DEBUG)
    target_compile_options(${MODULE}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # Device debug info
    )
endif()

add_executable(${MODULE}_long_term_test
    test_vector_add_2d.cu
)
target_link_libraries(${MODULE}_long_term_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
    CUDA::cudart
    ${MODULE_NAME}_lib_INTERFACE
)
gtest_discover_tests(${MODULE}_long_term_test)
add_profiling_targets(${MODULE}_long_term_test)