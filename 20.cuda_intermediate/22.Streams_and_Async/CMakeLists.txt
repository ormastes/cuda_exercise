cmake_minimum_required(VERSION 3.18)
project(22_Streams_And_Async CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(basic_streams basic_streams.cu)
target_link_libraries(basic_streams CUDA::cudart)

add_executable(overlap_transfer_compute overlap_transfer_compute.cu)
target_link_libraries(overlap_transfer_compute CUDA::cudart)

add_executable(multi_stream_pipeline multi_stream_pipeline.cu)
target_link_libraries(multi_stream_pipeline CUDA::cudart)

add_executable(stream_callback stream_callback.cu)
target_link_libraries(stream_callback CUDA::cudart Threads::Threads)