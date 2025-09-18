add_executable(cuda_unit_test
    host_test.cu
    device_test.cu
)

target_link_libraries(cuda_unit_test
    PRIVATE
    GTest::gtest_main
)

set_target_properties(cuda_unit_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

include(GoogleTest)
gtest_discover_tests(cuda_unit_test)