cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME ProjectName)
project(${ProjectName})

add_executable(${PROJECT_NAME} src.cu)