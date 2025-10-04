# Test executable with direct .cu inclusion
add_executable(${MODULE}_test
    unit/kernels/test_vector_add_2d.cu
)

# Link test with testing frameworks (direct inclusion, but with interface for headers)
target_link_libraries(${MODULE}_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
    ${MODULE}_lib_INTERFACE  # For clean header includes
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_test)

# Add profiling targets
add_profiling_targets(${MODULE}_test)

# Test executable with implementation library (part-specific)
add_executable(${MODULE}_library_test
    unit/part_specific/test_vector_add_2d_with_lib.cu
)

# Link test with implementation library and testing frameworks
target_link_libraries(${MODULE}_library_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
    ${MODULE}_lib
)

# Register tests with CTest
gtest_discover_tests(${MODULE}_library_test)

# Add profiling targets
add_profiling_targets(${MODULE}_library_test)

# Debug build specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${MODULE}_test PRIVATE DEBUG)
    target_compile_definitions(${MODULE}_library_test PRIVATE DEBUG)
    target_compile_options(${MODULE}_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # Device debug info
    )
    target_compile_options(${MODULE}_library_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # Device debug info
    )
endif()