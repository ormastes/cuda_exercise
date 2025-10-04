# Source library
add_library(${MODULE}_lib STATIC
    kernels/stream_kernels.cu
    kernels/async_kernels.cu
    utils/stream_manager.cu
)

target_compile_definitions(${MODULE}_lib PUBLIC BUILDING_LIB)

target_compile_options(${MODULE}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

set_target_properties(${MODULE}_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${MODULE}_lib PUBLIC CudaCustomLib)

# Example executables
add_executable(${MODULE}_pipeline examples/multi_stream_pipeline.cu)
target_link_libraries(${MODULE}_pipeline PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})

add_executable(${MODULE}_overlap examples/overlap_transfer_compute.cu)
target_link_libraries(${MODULE}_overlap PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})

add_executable(${MODULE}_callback examples/stream_callback.cu)
target_link_libraries(${MODULE}_callback PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})

add_executable(${MODULE}_graph examples/graph_api_demo.cu)
target_link_libraries(${MODULE}_graph PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})
