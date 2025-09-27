
project(14_CodeInspectionAndProfiling)

# Main example with NVTX, coalescing, and memory error demos
add_executable(${PROJECT_NAME} vector_add_2d.cu)
add_profiling_targets(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${CUDA_BASIC_LIB}
    CudaCustomLib  # Link our custom CUDA utilities library
    dl  # Required for NVTX dynamic loading
)

# Custom target for coalescing analysis
add_custom_target(${PROJECT_NAME}_coalescing_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Coalescing Patterns ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/coalescing_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo \"Running coalesced access...\" && \
            $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/coalesced_output.txt && \
            echo \"Running strided access...\" && \
            $<TARGET_FILE:${PROJECT_NAME}> --strided > $$PROFILE_DIR/strided_output.txt && \
            echo \"Analyzing memory efficiency...\" && \
            (ncu --metrics gld_efficiency,gst_efficiency \
                 --kernel-name \".*Coalesced\" \
                 --export $$PROFILE_DIR/coalesced_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/coalesced_ncu.txt 2>&1 || echo \"NCU analysis completed\") && \
            (ncu --metrics gld_efficiency,gst_efficiency \
                 --kernel-name \".*Strided\" \
                 --export $$PROFILE_DIR/strided_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> --strided > $$PROFILE_DIR/strided_ncu.txt 2>&1 || echo \"NCU analysis completed\") && \
            echo \"Coalescing analysis saved to: $$PROFILE_DIR\""
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory coalescing patterns"
)

# Custom target for NVTX timeline
add_custom_target(${PROJECT_NAME}_nvtx_timeline
    COMMAND ${CMAKE_COMMAND} -E echo "=== Generating NVTX Timeline ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/nvtx_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            nsys profile --stats=true --trace=cuda,nvtx,osrt -o $$PROFILE_DIR/timeline $<TARGET_FILE:${PROJECT_NAME}> && \
            nsys stats --report nvtx_sum $$PROFILE_DIR/timeline.nsys-rep > $$PROFILE_DIR/nvtx_summary.txt && \
            nsys stats --report cuda_kernel_sum $$PROFILE_DIR/timeline.nsys-rep > $$PROFILE_DIR/kernel_summary.txt && \
            echo \"NVTX timeline saved to: $$PROFILE_DIR/timeline.nsys-rep\" && \
            echo \"View with: nsys-ui $$PROFILE_DIR/timeline.nsys-rep\""
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Profiling NVTX markers and generating timeline"
)

# Custom target for memory error detection
add_custom_target(${PROJECT_NAME}_memory_error_check
    COMMAND ${CMAKE_COMMAND} -E echo "=== Checking for Memory Errors ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/memcheck_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo \"Running with intentional memory errors...\" && \
            compute-sanitizer --tool memcheck $<TARGET_FILE:${PROJECT_NAME}> --memory-error > $$PROFILE_DIR/memcheck.log 2>&1 || true && \
            echo \"Memory error detection saved to: $$PROFILE_DIR/memcheck.log\" && \
            grep -E \"ERROR|Invalid\" $$PROFILE_DIR/memcheck.log || echo \"Run with compute-sanitizer to see detailed errors\""
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Detecting memory errors with compute-sanitizer"
)
