project(15_Unit_Testing)
add_library(${PROJECT_NAME}_lib STATIC
    vector_add_2d.cu
)

# Create test executable
add_executable(${PROJECT_NAME}_test
    test_vector_add_2d.cu
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