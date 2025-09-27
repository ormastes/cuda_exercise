cmake_minimum_required(VERSION 3.18)
project(SharedMemory CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Remove the explicit architecture flag, let the parent CMakeLists handle it

add_executable(basic_shared_memory basic_shared_memory.cu)
target_link_libraries(basic_shared_memory CUDA::cudart)

add_executable(matrix_transpose matrix_transpose.cu)
target_link_libraries(matrix_transpose CUDA::cudart)

add_executable(reduction reduction.cu)
target_link_libraries(reduction CUDA::cudart)

add_executable(stencil_1d stencil_1d.cu)
target_link_libraries(stencil_1d CUDA::cudart)