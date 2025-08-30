cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME ProjectName)   

project(${ProjectName})

add_library(${PROJECT_NAME}_lib STATIC kernel.cu)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC CUDA::cudart)

add_executable(${PROJECT_NAME} src.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)


