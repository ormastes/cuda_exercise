project(17_Memory_Hierarchy)

# Main executable for memory hierarchy demonstrations
add_executable(${PROJECT_NAME} matrix_multiply.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Enable device-side debug symbols in debug mode
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# Create library for testing (exports kernels)
add_library(${PROJECT_NAME}_lib STATIC matrix_multiply.cu)

target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    BUILDING_LIB  # Define to avoid main() in library
)

# Unit tests
add_executable(${PROJECT_NAME}_test test_matrix_multiply.cu)

target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_lib
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
    CUDA::cudart
)

# Register tests with CTest
gtest_discover_tests(${PROJECT_NAME}_test)

# Profiling targets
add_profiling_targets(${PROJECT_NAME})

# Custom target for memory pattern analysis
add_custom_target(${PROJECT_NAME}_memory_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Access Patterns ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/memory_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Running memory hierarchy benchmark...' && \
            $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/output.txt && \
            echo 'Profiling with Nsight Compute...' && \
            (ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_data_bytes_mem_shared_ld.sum,\
smsp__sass_data_bytes_mem_shared_st.sum \
                 --kernel-name matmul \
                 --export $$PROFILE_DIR/memory_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/ncu_output.txt 2>&1 || echo 'NCU analysis complete') && \
            echo 'Memory analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory access patterns and efficiency"
)

# Custom target for shared memory bank conflict analysis
add_custom_target(${PROJECT_NAME}_bank_conflict_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Shared Memory Bank Conflicts ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/bank_conflicts_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Profiling shared memory kernels...' && \
            (ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
                 --kernel-name matmul_shared \
                 --export $$PROFILE_DIR/shared_conflicts \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/shared_conflicts.txt 2>&1 || true) && \
            (ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
                 --kernel-name matmul_shared_bank_conflict_free \
                 --export $$PROFILE_DIR/conflict_free \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/conflict_free.txt 2>&1 || true) && \
            echo 'Bank conflict analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing shared memory bank conflicts"
)

# Custom target for performance comparison
add_custom_target(${PROJECT_NAME}_performance_comparison
    COMMAND ${CMAKE_COMMAND} -E echo "=== Performance Comparison ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/perf_comparison_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Testing different matrix sizes...' && \
            for size in 128 256 512 1024; do \
                echo \"\\n=== Matrix Size: $$size x $$size ===\" | tee -a $$PROFILE_DIR/performance.txt && \
                $<TARGET_FILE:${PROJECT_NAME}> $$size | tee -a $$PROFILE_DIR/performance.txt; \
            done && \
            echo 'Performance comparison saved to: $$PROFILE_DIR/performance.txt'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance comparison across different matrix sizes"
)

# Custom target for cache analysis
add_custom_target(${PROJECT_NAME}_cache_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Cache Performance Analysis ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/cache_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Analyzing L1/L2 cache performance...' && \
            (ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
lts__t_sectors_op_read_lookup_miss.sum \
                 --kernel-name matmul \
                 --export $$PROFILE_DIR/cache_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 512 > $$PROFILE_DIR/cache_output.txt 2>&1 || echo 'Cache analysis complete') && \
            echo 'Cache analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing L1/L2 cache hit rates and performance"
)

# Add test for the main executable
if(BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_main_test
        COMMAND ${PROJECT_NAME} 64
    )
    set_tests_properties(${PROJECT_NAME}_main_test PROPERTIES
        PASS_REGULAR_EXPRESSION "Memory Hierarchy Demo Complete"
        TIMEOUT 30
    )
endif()