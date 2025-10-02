project(15_Unit_Testing)
add_library(${PROJECT_NAME}_lib STATIC
    vector_add_2d.cu
)
# Enable CUDA separable compilation for test executable
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Create test executable
add_executable(${PROJECT_NAME}_test
    test_vector_add_2d.cu
)
# Enable CUDA separable compilation for test executable
set_target_properties(${PROJECT_NAME}_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Link test with implementation library and testing frameworks
target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_lib
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib  # Link our custom CUDA utilities library
)

gtest_discover_tests(${PROJECT_NAME}_test)