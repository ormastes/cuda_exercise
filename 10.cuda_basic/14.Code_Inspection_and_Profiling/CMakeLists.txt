
project(14_Code_Inspection_And_Profiling)

# Main example with NVTX, coalescing, and memory error demos
add_executable(${PROJECT_NAME} vector_add_2d.cu)
add_profiling_targets(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${CUDA_BASIC_LIB}
    CudaCustomLib  # Link our custom CUDA utilities library
    ${NVTX_PLATFORM_LIBS} # Link platform-specific libraries if needed

)

# Memory errors demonstration executable
add_executable(memory_errors memory_errors.cu)
target_link_libraries(memory_errors
PRIVATE
    ${CUDA_BASIC_LIB}
)

# Custom target for coalescing analysis - simplified to avoid shell syntax issues
add_custom_target(${PROJECT_NAME}_coalescing_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Coalescing Patterns ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen/coalescing_analysis
    COMMAND $<TARGET_FILE:${PROJECT_NAME}> > ${CMAKE_BINARY_DIR}/gen/coalescing_analysis/coalesced_output.txt
    COMMAND $<TARGET_FILE:${PROJECT_NAME}> --strided > ${CMAKE_BINARY_DIR}/gen/coalescing_analysis/strided_output.txt
    COMMAND $<TARGET_FILE:${PROJECT_NAME}> --compare > ${CMAKE_BINARY_DIR}/gen/coalescing_analysis/compare_output.txt
    COMMAND ${CMAKE_COMMAND} -E echo "Analysis saved to: ${CMAKE_BINARY_DIR}/gen/coalescing_analysis/"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory coalescing patterns"
)

# Custom target for NVTX timeline - simplified
add_custom_target(${PROJECT_NAME}_nvtx_timeline
    COMMAND ${CMAKE_COMMAND} -E echo "=== Generating NVTX Timeline ==="
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/gen/nvtx_timeline
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen/nvtx_timeline
    COMMAND nsys profile --stats=true --trace=cuda,nvtx,osrt --force-overwrite=true -o ${CMAKE_BINARY_DIR}/gen/nvtx_timeline/timeline $<TARGET_FILE:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E echo "NVTX timeline saved to: ${CMAKE_BINARY_DIR}/gen/nvtx_timeline/timeline.nsys-rep"
    COMMAND ${CMAKE_COMMAND} -E echo "View with: nsys-ui ${CMAKE_BINARY_DIR}/gen/nvtx_timeline/timeline.nsys-rep"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Profiling NVTX markers and generating timeline"
)

# Custom target for memory error detection with vector_add_2d - simplified
add_custom_target(${PROJECT_NAME}_memory_error_check
    COMMAND ${CMAKE_COMMAND} -E echo "=== Checking for Memory Errors ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen/memcheck
    COMMAND ${CMAKE_COMMAND} -E echo "Running with intentional memory errors..."
    COMMAND compute-sanitizer --tool memcheck --save ${CMAKE_BINARY_DIR}/gen/memcheck/memcheck.log $<TARGET_FILE:${PROJECT_NAME}> --memory-error > ${CMAKE_BINARY_DIR}/gen/memcheck/output.txt 2>&1 || true
    COMMAND ${CMAKE_COMMAND} -E echo "Memory error detection saved to: ${CMAKE_BINARY_DIR}/gen/memcheck/"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Detecting memory errors with compute-sanitizer"
)

# Custom targets for comprehensive memory_errors testing - simplified
add_custom_target(memory_errors_all_checks
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running All Sanitizer Checks on memory_errors ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen/memory_errors_all
    COMMAND ${CMAKE_COMMAND} -E echo "1. Memory check - detecting out-of-bounds access..."
    COMMAND compute-sanitizer --tool memcheck --log-file ${CMAKE_BINARY_DIR}/gen/memory_errors_all/memcheck_output.txt $<TARGET_FILE:memory_errors>
    COMMAND ${CMAKE_COMMAND} -E echo "2. Race check - detecting race conditions..."
    COMMAND compute-sanitizer --tool racecheck --log-file ${CMAKE_BINARY_DIR}/gen/memory_errors_all/racecheck_output.txt $<TARGET_FILE:memory_errors>
    COMMAND ${CMAKE_COMMAND} -E echo "3. Init check - detecting uninitialized memory..."
    COMMAND compute-sanitizer --tool initcheck --log-file ${CMAKE_BINARY_DIR}/gen/memory_errors_all/initcheck_output.txt $<TARGET_FILE:memory_errors>
    COMMAND ${CMAKE_COMMAND} -E echo "Full reports saved to: ${CMAKE_BINARY_DIR}/gen/memory_errors_all/"
    DEPENDS memory_errors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all compute-sanitizer checks on memory_errors example"
)

# Individual sanitizer targets for memory_errors
add_custom_target(memory_errors_memcheck
    COMMAND compute-sanitizer --tool memcheck $<TARGET_FILE:memory_errors>
    DEPENDS memory_errors
    COMMENT "Running memcheck on memory_errors"
)

add_custom_target(memory_errors_racecheck
    COMMAND compute-sanitizer --tool racecheck $<TARGET_FILE:memory_errors>
    DEPENDS memory_errors
    COMMENT "Running racecheck on memory_errors"
)

add_custom_target(memory_errors_initcheck
    COMMAND compute-sanitizer --tool initcheck $<TARGET_FILE:memory_errors>
    DEPENDS memory_errors
    COMMENT "Running initcheck on memory_errors"
)
