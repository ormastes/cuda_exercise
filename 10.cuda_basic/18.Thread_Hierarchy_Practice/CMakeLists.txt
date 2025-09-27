project(18_Thread_Hierarchy_Practice)

# Main executable for thread hierarchy demonstrations
add_executable(${PROJECT_NAME} matrix_multiply_tiled.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Enable device-side debug symbols in debug mode
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# Create library for testing (exports kernels)
add_library(${PROJECT_NAME}_lib STATIC matrix_multiply_tiled.cu)

target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    BUILDING_LIB  # Define to avoid main() in library
)

# Unit tests
add_executable(${PROJECT_NAME}_test test_matrix_multiply_tiled.cu)

target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_lib
    GTest::gtest_main
    CUDA::cudart
)

# Register tests with CTest
gtest_discover_tests(${PROJECT_NAME}_test)

# Profiling targets
add_profiling_targets(${PROJECT_NAME})

# Custom target for thread hierarchy analysis
add_custom_target(${PROJECT_NAME}_thread_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Thread Hierarchy Configurations ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/thread_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Running thread hierarchy benchmark...' && \
            $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/output.txt && \
            echo 'Profiling different tile configurations...' && \
            (ncu --metrics sm__threads_launched.sum,\
sm__warps_launched.sum,\
sm__maximum_warps_per_active_cycle_pct,\
sm__occupancy_target_register_per_thread \
                 --kernel-name matmul_tiled \
                 --export $$PROFILE_DIR/thread_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/ncu_output.txt 2>&1 || echo 'NCU analysis complete') && \
            echo 'Thread analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing thread hierarchy configurations and efficiency"
)

# Custom target for warp divergence analysis
add_custom_target(${PROJECT_NAME}_divergence_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Warp Divergence ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/divergence_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Profiling warp divergence...' && \
            (ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,\
smsp__branch_targets_threads_divergent.pct,\
smsp__warps_launched.sum \
                 --kernel-name demonstrate \
                 --export $$PROFILE_DIR/divergence_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/divergence_output.txt 2>&1 || true) && \
            echo 'Divergence analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing warp divergence patterns"
)

# Custom target for occupancy analysis
add_custom_target(${PROJECT_NAME}_occupancy_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Occupancy Analysis ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/occupancy_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Testing occupancy with different block sizes...' && \
            $<TARGET_FILE:${PROJECT_NAME}> 512 > $$PROFILE_DIR/occupancy.txt && \
            echo 'Profiling achieved occupancy...' && \
            (ncu --metrics sm__maximum_warps_per_active_cycle_pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_blocks,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem \
                 --kernel-name matmul \
                 --export $$PROFILE_DIR/occupancy_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 512 > $$PROFILE_DIR/ncu_occupancy.txt 2>&1 || echo 'Occupancy analysis complete') && \
            echo 'Occupancy analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing kernel occupancy and limiting factors"
)

# Custom target for tile size comparison
add_custom_target(${PROJECT_NAME}_tile_comparison
    COMMAND ${CMAKE_COMMAND} -E echo "=== Tile Size Performance Comparison ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/tile_comparison_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Testing different tile sizes...' && \
            for size in 128 256 512 1024; do \
                echo \"\\n=== Matrix Size: $$size x $$size ===\" | tee -a $$PROFILE_DIR/performance.txt && \
                $<TARGET_FILE:${PROJECT_NAME}> $$size | grep -E 'Tiled|Rectangular|Warp' | tee -a $$PROFILE_DIR/performance.txt; \
            done && \
            echo 'Tile comparison saved to: $$PROFILE_DIR/performance.txt'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Comparing performance of different tile configurations"
)

# Custom target for thread coarsening analysis
add_custom_target(${PROJECT_NAME}_coarsening_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Thread Coarsening Analysis ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/coarsening_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Analyzing thread coarsening benefits...' && \
            (ncu --metrics smsp__inst_executed.sum,\
smsp__thread_inst_executed.sum,\
sm__inst_executed_pipe_lsu_per_cycle.avg.pct_of_peak_sustained_elapsed \
                 --kernel-name matmul_thread_coarsening \
                 --export $$PROFILE_DIR/coarsening_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> 256 > $$PROFILE_DIR/coarsening_output.txt 2>&1 || true) && \
            echo 'Coarsening analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing thread coarsening effects on performance"
)

# Add test for the main executable
if(BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_main_test
        COMMAND ${PROJECT_NAME} 64
    )
    set_tests_properties(${PROJECT_NAME}_main_test PROPERTIES
        PASS_REGULAR_EXPRESSION "Thread Hierarchy Demo Complete"
        TIMEOUT 30
    )
endif()