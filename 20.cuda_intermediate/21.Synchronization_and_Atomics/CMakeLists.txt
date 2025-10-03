project(22_Synchronization_and_Atomics)

# Main executable for synchronization and atomics demonstrations
add_executable(${PROJECT_NAME} synchronization_atomics.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Enable cooperative groups
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create library for testing
add_library(${PROJECT_NAME}_lib STATIC synchronization_atomics.cu)

target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    BUILDING_LIB
)

target_compile_options(${PROJECT_NAME}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

set_target_properties(${PROJECT_NAME}_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Unit tests
add_executable(${PROJECT_NAME}_test test_synchronization_atomics.cu)

target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    ${PROJECT_NAME}_lib
    GTest::gtest_main
    CUDA::cudart
)

target_compile_options(${PROJECT_NAME}_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

set_target_properties(${PROJECT_NAME}_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Register tests with CTest
gtest_discover_tests(${PROJECT_NAME}_test)

# Profiling targets
add_profiling_targets(${PROJECT_NAME})

# Custom target for atomic performance analysis
add_custom_target(${PROJECT_NAME}_atomic_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Atomic Operations Performance ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/atomic_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Running atomic operations benchmark...' && \
            $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/output.txt && \
            echo 'Profiling atomic operations...' && \
            (ncu --metrics sm__inst_executed_op_global_red.sum,\
sm__inst_executed_op_shared_atom.sum,\
sm__inst_executed_op_global_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum \
                 --kernel-name atomic \
                 --export $$PROFILE_DIR/atomic_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/ncu_output.txt 2>&1 || echo 'NCU analysis complete') && \
            echo 'Atomic analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing atomic operations performance and contention"
)

# Custom target for synchronization overhead analysis
add_custom_target(${PROJECT_NAME}_sync_overhead
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Synchronization Overhead ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/sync_overhead_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Measuring synchronization overhead...' && \
            (nsys profile --stats=true \
                 --output=$$PROFILE_DIR/sync_profile \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/nsys_output.txt 2>&1 || echo 'NSYS profiling complete') && \
            echo 'Synchronization analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Measuring synchronization barrier overhead"
)

# Custom target for memory fence analysis
add_custom_target(${PROJECT_NAME}_fence_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Fence Impact ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/fence_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Analyzing memory fence operations...' && \
            (ncu --metrics smsp__inst_executed_op_fence.sum,\
smsp__inst_executed_op_bar.sum,\
gpu__time_duration.sum \
                 --kernel-name fence \
                 --export $$PROFILE_DIR/fence_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/ncu_output.txt 2>&1 || echo 'Fence analysis complete') && \
            echo 'Fence analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory fence performance impact"
)

# Custom target for cooperative groups analysis
add_custom_target(${PROJECT_NAME}_coop_groups_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Cooperative Groups ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/coop_groups_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Testing cooperative groups support...' && \
            $<TARGET_FILE:${PROJECT_NAME}> | grep -A 10 'Cooperative Groups' > $$PROFILE_DIR/coop_output.txt && \
            echo 'Cooperative groups analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing cooperative groups capabilities"
)

# Custom target for lock contention analysis
add_custom_target(${PROJECT_NAME}_lock_contention
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Lock Contention ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/lock_contention_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Measuring lock contention...' && \
            $<TARGET_FILE:${PROJECT_NAME}> | grep -A 10 'Spinlock' > $$PROFILE_DIR/lock_output.txt && \
            (ncu --metrics sm__warps_active.avg,\
sm__warps_active.max,\
smsp__warps_cant_issue_membar.sum \
                 --kernel-name spinlock \
                 --export $$PROFILE_DIR/lock_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/ncu_lock.txt 2>&1 || echo 'Lock analysis complete') && \
            echo 'Lock contention analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Measuring spinlock contention and performance"
)

# Add test for the main executable
if(BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_main_test
        COMMAND ${PROJECT_NAME}
    )
    set_tests_properties(${PROJECT_NAME}_main_test PROPERTIES
        PASS_REGULAR_EXPRESSION "Synchronization and Atomics Demo Complete"
        TIMEOUT 30
    )
endif()