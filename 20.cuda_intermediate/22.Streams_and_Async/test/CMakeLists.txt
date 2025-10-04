# Test executable
add_executable(${MODULE}_test
    kernels/test_stream_kernels.cu
    part_specific/test_stream_synchronization.cu
    part_specific/test_async_operations.cu
    part_specific/benchmark_streams.cu
)

target_include_directories(${MODULE}_test PRIVATE
    ${CMAKE_SOURCE_DIR}/00.test_lib
)

target_link_libraries(${MODULE}_test PRIVATE
    ${MODULE}_lib
    ${CUDA_BASIC_LIB}
    ${GTEST_BASIC_LIB}
)

gtest_discover_tests(${MODULE}_test)
add_profiling_targets(${MODULE}_test)
