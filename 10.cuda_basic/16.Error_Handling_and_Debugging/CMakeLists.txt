project(16_ErrorHandlingAndDebugging)

# Main error handling demonstration with vector_add_2d kernels
add_executable(${PROJECT_NAME}
    error_handling_demo.cu
    vector_add_2d.cu
)

# Link libraries including NVTX for profiling
target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${CUDA_BASIC_LIB}
    CudaCustomLib  # Link our custom CUDA utilities library if available
)

# Add profiling targets if enabled
add_profiling_targets(${PROJECT_NAME})

# Custom target for sanitizer testing
add_custom_target(${PROJECT_NAME}_sanitize_all
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running Comprehensive Sanitizer Checks ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/sanitize_${PROJECT_NAME}_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo '--- Memory Check ---' && \
            compute-sanitizer --tool memcheck --print-level info $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/memcheck.log 2>&1 || true && \
            echo '--- Race Check ---' && \
            compute-sanitizer --tool racecheck --print-level info $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/racecheck.log 2>&1 || true && \
            echo '--- Sync Check ---' && \
            compute-sanitizer --tool synccheck $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/synccheck.log 2>&1 || true && \
            echo '--- Init Check ---' && \
            compute-sanitizer --tool initcheck $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/initcheck.log 2>&1 || true && \
            echo '' && \
            echo '=== Sanitizer Summary ===' && \
            echo 'Memory errors:' && grep -c ERROR $$PROFILE_DIR/memcheck.log 2>/dev/null || echo '0' && \
            echo 'Race conditions:' && grep -c 'race' $$PROFILE_DIR/racecheck.log 2>/dev/null || echo '0' && \
            echo 'Sync issues:' && grep -c 'error' $$PROFILE_DIR/synccheck.log 2>/dev/null || echo '0' && \
            echo '' && \
            echo 'Full results saved to: $$PROFILE_DIR/'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all compute-sanitizer checks on ${PROJECT_NAME}"
)

# Debug build specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>  # Device debug info
    )
endif()

# Add test for the error handling demo
if(BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_test
        COMMAND ${PROJECT_NAME}
    )
    set_tests_properties(${PROJECT_NAME}_test PROPERTIES
        PASS_REGULAR_EXPRESSION "CUDA Error Handling and Debugging Demonstration"
        TIMEOUT 30
    )
endif()