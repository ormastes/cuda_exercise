
project(14_CodeInspectionAndProfiling)

# Main example
add_executable(${PROJECT_NAME} vector_add_2d.cu)
add_profiling_targets(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${CUDA_BASIC_LIB}
)

# Memory errors example for sanitizer testing
add_executable(${PROJECT_NAME}_memory_errors memory_errors.cu)
target_link_libraries(${PROJECT_NAME}_memory_errors PRIVATE ${CUDA_BASIC_LIB})
add_profiling_targets(${PROJECT_NAME}_memory_errors)

# Add custom sanitizer targets for memory_errors
add_custom_target(${PROJECT_NAME}_memory_errors_memcheck
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/memcheck_$$(date +%Y%m%d_%H%M%S)_memory_errors && \
            mkdir -p $$PROFILE_DIR && \
            compute-sanitizer --tool memcheck --log-file $$PROFILE_DIR/memcheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> | tee $$PROFILE_DIR/output.txt && \
            echo \"Memcheck results saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_memory_errors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running memory check on memory_errors"
)

add_custom_target(${PROJECT_NAME}_memory_errors_racecheck
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/racecheck_$$(date +%Y%m%d_%H%M%S)_memory_errors && \
            mkdir -p $$PROFILE_DIR && \
            compute-sanitizer --tool racecheck --log-file $$PROFILE_DIR/racecheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> | tee $$PROFILE_DIR/output.txt && \
            echo \"Racecheck results saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_memory_errors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running race condition check on memory_errors"
)

add_custom_target(${PROJECT_NAME}_memory_errors_synccheck
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/synccheck_$$(date +%Y%m%d_%H%M%S)_memory_errors && \
            mkdir -p $$PROFILE_DIR && \
            compute-sanitizer --tool synccheck --log-file $$PROFILE_DIR/synccheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> | tee $$PROFILE_DIR/output.txt && \
            echo \"Synccheck results saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_memory_errors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running synchronization check on memory_errors"
)

# Combined sanitizer target
add_custom_target(${PROJECT_NAME}_memory_errors_all_sanitizers
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/all_sanitizers_$$(date +%Y%m%d_%H%M%S)_memory_errors && \
            mkdir -p $$PROFILE_DIR && \
            echo \"Running all sanitizer checks...\" && \
            compute-sanitizer --tool memcheck --log-file $$PROFILE_DIR/memcheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> && \
            compute-sanitizer --tool racecheck --log-file $$PROFILE_DIR/racecheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> && \
            compute-sanitizer --tool synccheck --log-file $$PROFILE_DIR/synccheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> && \
            compute-sanitizer --tool initcheck --log-file $$PROFILE_DIR/initcheck.log $<TARGET_FILE:${PROJECT_NAME}_memory_errors> && \
            echo \"All sanitizer results saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_memory_errors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all sanitizer checks on memory_errors"
)

# NVTX example
add_executable(${PROJECT_NAME}_nvtx_example nvtx_example.cu)
target_link_libraries(${PROJECT_NAME}_nvtx_example PRIVATE ${CUDA_BASIC_LIB} CUDA::nvtx3)
add_profiling_targets(${PROJECT_NAME}_nvtx_example)

# Add custom NVTX profiling target
add_custom_target(${PROJECT_NAME}_nvtx_example_timeline
    COMMAND ${CMAKE_COMMAND} -E echo "=== Generating NVTX Timeline ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/nvtx_timeline_$$(date +%Y%m%d_%H%M%S)_nvtx_example && \
            mkdir -p $$PROFILE_DIR && \
            nsys profile --stats=true --trace=cuda,nvtx,osrt -o $$PROFILE_DIR/timeline $<TARGET_FILE:${PROJECT_NAME}_nvtx_example> && \
            nsys stats --report nvtx_sum $$PROFILE_DIR/timeline.nsys-rep > $$PROFILE_DIR/nvtx_summary.txt && \
            nsys stats --report cuda_kernel_sum $$PROFILE_DIR/timeline.nsys-rep > $$PROFILE_DIR/kernel_summary.txt && \
            echo \"NVTX timeline saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_nvtx_example
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Profiling NVTX markers and generating timeline"
)

# Memory coalescing example
add_executable(${PROJECT_NAME}_memory_coalescing memory_coalescing.cu)
target_link_libraries(${PROJECT_NAME}_memory_coalescing PRIVATE ${CUDA_BASIC_LIB})
add_profiling_targets(${PROJECT_NAME}_memory_coalescing)

# Add custom memory analysis targets
add_custom_target(${PROJECT_NAME}_memory_coalescing_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Access Patterns ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/memory_analysis_$$(date +%Y%m%d_%H%M%S)_memory_coalescing && \
            mkdir -p $$PROFILE_DIR && \
            echo \"Running program...\" && \
            $<TARGET_FILE:${PROJECT_NAME}_memory_coalescing> > $$PROFILE_DIR/output.txt && \
            echo \"Analyzing memory efficiency...\" && \
            (ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions \
                 --kernel-name \".*coalescedAccess.*\" \
                 --export $$PROFILE_DIR/coalesced_metrics \
                 --print-summary per-kernel \
                 $<TARGET_FILE:${PROJECT_NAME}_memory_coalescing> > $$PROFILE_DIR/coalesced_analysis.txt 2>&1 || echo \"NCU analysis completed\") && \
            (ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions \
                 --kernel-name \".*uncoalescedAccess.*\" \
                 --export $$PROFILE_DIR/uncoalesced_metrics \
                 --print-summary per-kernel \
                 $<TARGET_FILE:${PROJECT_NAME}_memory_coalescing> > $$PROFILE_DIR/uncoalesced_analysis.txt 2>&1 || echo \"NCU analysis completed\") && \
            echo \"Memory analysis saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}_memory_coalescing
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory coalescing patterns"
)

# Combined analysis target for all examples
add_custom_target(${PROJECT_NAME}_profile_all_examples
    COMMAND ${CMAKE_COMMAND} --build . --target ${PROJECT_NAME}_memory_errors_profile
    COMMAND ${CMAKE_COMMAND} --build . --target ${PROJECT_NAME}_nvtx_example_timeline
    COMMAND ${CMAKE_COMMAND} --build . --target ${PROJECT_NAME}_memory_coalescing_analysis
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all profiling examples"
)
