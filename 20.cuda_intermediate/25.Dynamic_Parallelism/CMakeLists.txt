project(25_DynamicParallelism)

# Note: Dynamic parallelism requires special compilation flags set in parent CMakeLists.txt
# Enable relocatable device code for this project
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Quicksort example
add_executable(${PROJECT_NAME}_quicksort quicksort.cu)
set_target_properties(${PROJECT_NAME}_quicksort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_profiling_targets(${PROJECT_NAME}_quicksort)
target_link_libraries(${PROJECT_NAME}_quicksort
    PRIVATE
    ${CUDA_BASIC_LIB}
    CUDA::cudadevrt  # Required for dynamic parallelism
)

# Tree traversal example
add_executable(${PROJECT_NAME}_tree_traversal tree_traversal.cu)
set_target_properties(${PROJECT_NAME}_tree_traversal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_profiling_targets(${PROJECT_NAME}_tree_traversal)
target_link_libraries(${PROJECT_NAME}_tree_traversal
    PRIVATE
    ${CUDA_BASIC_LIB}
    CUDA::cudadevrt
)

# Adaptive mesh refinement
add_executable(${PROJECT_NAME}_adaptive_mesh adaptive_mesh.cu)
set_target_properties(${PROJECT_NAME}_adaptive_mesh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_profiling_targets(${PROJECT_NAME}_adaptive_mesh)
target_link_libraries(${PROJECT_NAME}_adaptive_mesh
    PRIVATE
    ${CUDA_BASIC_LIB}
    CUDA::cudadevrt
)

# Recursive matrix multiplication
add_executable(${PROJECT_NAME}_recursive_matmul recursive_matmul.cu)
set_target_properties(${PROJECT_NAME}_recursive_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_profiling_targets(${PROJECT_NAME}_recursive_matmul)
target_link_libraries(${PROJECT_NAME}_recursive_matmul
    PRIVATE
    ${CUDA_BASIC_LIB}
    CUDA::cudadevrt
)

# Adaptive integration
add_executable(${PROJECT_NAME}_adaptive_integration adaptive_integration.cu)
set_target_properties(${PROJECT_NAME}_adaptive_integration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_profiling_targets(${PROJECT_NAME}_adaptive_integration)
target_link_libraries(${PROJECT_NAME}_adaptive_integration
    PRIVATE
    ${CUDA_BASIC_LIB}
    CUDA::cudadevrt
)