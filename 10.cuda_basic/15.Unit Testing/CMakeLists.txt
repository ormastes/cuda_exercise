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

# Add parameterized test executables
add_executable(test_gpu_parameterized test_gpu_parameterized.cu)
add_executable(test_gpu_generator_simple test_gpu_generator_simple.cu)
add_executable(test_gpu_generator test_gpu_generator.cu)

# Link libraries for parameterized tests
target_link_libraries(test_gpu_parameterized
    PRIVATE
    GTest::gtest_main
)

target_link_libraries(test_gpu_generator_simple
    PRIVATE
    GTest::gtest_main
)

target_link_libraries(test_gpu_generator
    PRIVATE
    GTest::gtest_main
)

# Set properties for parameterized tests
set_target_properties(test_gpu_parameterized PROPERTIES
    CUDA_ARCHITECTURES native
)

set_target_properties(test_gpu_generator_simple PROPERTIES
    CUDA_ARCHITECTURES native
)

set_target_properties(test_gpu_generator PROPERTIES
    CUDA_ARCHITECTURES native
)

# Add parameterized tests
gtest_discover_tests(test_gpu_parameterized)
gtest_discover_tests(test_gpu_generator_simple)
# Disable test discovery for test_gpu_generator due to runtime issues during build
# gtest_discover_tests(test_gpu_generator)
add_test(NAME test_gpu_generator COMMAND test_gpu_generator)