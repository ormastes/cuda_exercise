cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME ProjectName)
project(${ProjectName})

add_library(${PROJECT_NAME}_lib STATIC kernel.cu)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC CUDA::cudart)

add_executable(${PROJECT_NAME} src.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# Tests
add_executable(test_${PROJECT_NAME} test_kernel.cpp)
target_link_libraries(test_${PROJECT_NAME} PRIVATE 
    ${PROJECT_NAME}_lib
    Catch2::Catch2WithMain
)

# Register test with CTest
catch_discover_tests(test_${PROJECT_NAME})
