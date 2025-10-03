project(24_Memory_Coalescing_And_Bank_Conflicts)

# Coalescing demonstration
add_executable(${PROJECT_NAME}_coalescing_demo coalescing_demo.cu)
add_profiling_targets(${PROJECT_NAME}_coalescing_demo)
add_coalescing_analysis(${PROJECT_NAME}_coalescing_demo)
target_link_libraries(${PROJECT_NAME}_coalescing_demo
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# Bank conflicts demonstration
add_executable(${PROJECT_NAME}_bank_conflicts bank_conflicts.cu)
add_profiling_targets(${PROJECT_NAME}_bank_conflicts)
target_link_libraries(${PROJECT_NAME}_bank_conflicts
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# AoS vs SoA comparison
add_executable(${PROJECT_NAME}_aos_vs_soa aos_vs_soa.cu)
add_profiling_targets(${PROJECT_NAME}_aos_vs_soa)
target_link_libraries(${PROJECT_NAME}_aos_vs_soa
    PRIVATE
    ${CUDA_BASIC_LIB}
)

# Memory access patterns analysis
add_executable(${PROJECT_NAME}_memory_patterns memory_patterns.cu)
add_profiling_targets(${PROJECT_NAME}_memory_patterns)
add_coalescing_analysis(${PROJECT_NAME}_memory_patterns)
target_link_libraries(${PROJECT_NAME}_memory_patterns
    PRIVATE
    ${CUDA_BASIC_LIB}
)