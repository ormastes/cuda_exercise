project(23_Shared_Memory)

# Matrix transpose example
add_executable(${PROJECT_NAME}_matrix_transpose matrix_transpose.cu)
add_profiling_targets(${PROJECT_NAME}_matrix_transpose)
target_link_libraries(${PROJECT_NAME}_matrix_transpose
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# 1D Convolution example
add_executable(${PROJECT_NAME}_convolution_1d convolution_1d.cu)
add_profiling_targets(${PROJECT_NAME}_convolution_1d)
target_link_libraries(${PROJECT_NAME}_convolution_1d
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# Parallel reduction example
add_executable(${PROJECT_NAME}_reduction reduction.cu)
add_profiling_targets(${PROJECT_NAME}_reduction)
target_link_libraries(${PROJECT_NAME}_reduction
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# Stencil computation example
add_executable(${PROJECT_NAME}_stencil_1d stencil_1d.cu)
add_profiling_targets(${PROJECT_NAME}_stencil_1d)
target_link_libraries(${PROJECT_NAME}_stencil_1d
    PRIVATE
    ${CUDA_BASIC_LIB}
)