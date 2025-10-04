# Test executable
add_executable(${MODULE}_test
    kernels/test_atomics.cu
    kernels/test_synchronization.cu
    part_specific/test_lock_free.cu
    part_specific/benchmark_atomics.cu
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
