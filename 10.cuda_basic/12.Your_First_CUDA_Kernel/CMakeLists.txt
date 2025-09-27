
project(12_Your_First_CUDA_Kernel)

add_executable(${PROJECT_NAME} vector_add_2d.cu)

target_link_libraries(${PROJECT_NAME} CUDA::cudart)