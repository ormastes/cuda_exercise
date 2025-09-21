# inspect_profile.cmake - CUDA profiling and inspection utilities
# Provides functions for profiling, sanitization, and performance analysis

# Function to add comprehensive profiling targets for CUDA executables
# Usage: add_profiling_targets(target_name)
# Creates: ${target_name}_profile target that runs Nsight Systems, Nsight Compute, and Compute Sanitizer
function(add_profiling_targets TARGET_NAME)
    if(ENABLE_PROFILING)
        if(NOT TARGET ${TARGET_NAME})
            message(WARNING "Target ${TARGET_NAME} does not exist")
            return()
        endif()

        # Ensure NVTX is linked for profiling
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::nvtx3)

        # Single combined target for all profiling and checking - continues on errors
        add_custom_target(${TARGET_NAME}_profile
            COMMAND ${CMAKE_COMMAND} -E echo "=== Creating output directory ==="
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    echo \"Profile output directory: $$PROFILE_DIR\" && \
                    echo \"=== Nsight Systems Profiling ===\" && \
                    (nsys profile -o $$PROFILE_DIR/nsys_report --stats=true -t cuda,nvtx,osrt $<TARGET_FILE:${TARGET_NAME}> || echo \"Nsight Systems completed or failed\") && \
                    echo \"=== Nsight Compute Profiling ===\" && \
                    echo \"Note: If you see ERR_NVGPUCTRPERM, run: ./setup_ncu_permissions.sh\" && \
                    (ncu --set full -o $$PROFILE_DIR/ncu_report $<TARGET_FILE:${TARGET_NAME}> || echo \"Nsight Compute skipped - may need permissions\") && \
                    echo \"=== Compute Sanitizer All Checks: synccheck, racecheck, memcheck===\" && \
                    (compute-sanitizer $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/sanitizer_report.txt 2>&1 || echo \"Compute sanitizer completed or failed\") && \
                    echo \"=== Profile complete. Results saved in: $$PROFILE_DIR ===\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running complete profiling and checking suite on ${TARGET_NAME} (continues on errors)"
        )

        # Individual profiling targets for more focused analysis

        # Nsight Systems only
        add_custom_target(${TARGET_NAME}_nsys
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/nsys_${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    nsys profile -o $$PROFILE_DIR/timeline --stats=true -t cuda,nvtx,osrt $<TARGET_FILE:${TARGET_NAME}> && \
                    nsys stats --report nvtx_sum $$PROFILE_DIR/timeline.nsys-rep > $$PROFILE_DIR/nvtx_summary.txt || true && \
                    echo \"Nsight Systems timeline saved to: $$PROFILE_DIR/timeline.nsys-rep\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Profiling ${TARGET_NAME} with Nsight Systems"
        )

        # Nsight Compute only
        add_custom_target(${TARGET_NAME}_ncu
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/ncu_${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    echo \"Note: If you see ERR_NVGPUCTRPERM, run: ./setup_ncu_permissions.sh\" && \
                    (ncu --set full -o $$PROFILE_DIR/kernel_analysis $<TARGET_FILE:${TARGET_NAME}> || echo \"NCU may need permissions\") && \
                    echo \"Nsight Compute results saved to: $$PROFILE_DIR/kernel_analysis.ncu-rep\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Analyzing ${TARGET_NAME} kernels with Nsight Compute"
        )

        # Compute Sanitizer only
        add_custom_target(${TARGET_NAME}_sanitize
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/sanitize_${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    compute-sanitizer --tool memcheck $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/memcheck.log 2>&1 || true && \
                    compute-sanitizer --tool racecheck $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/racecheck.log 2>&1 || true && \
                    compute-sanitizer --tool synccheck $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/synccheck.log 2>&1 || true && \
                    echo \"Sanitizer results saved to: $$PROFILE_DIR/\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running sanitizers on ${TARGET_NAME}"
        )
    else()
        # ENABLE_PROFILING is OFF - create dummy function that does nothing
        message(STATUS "Profiling disabled for ${TARGET_NAME} (set ENABLE_PROFILING=ON to enable)")
    endif()
endfunction()

# Function to add memory coalescing analysis targets
function(add_coalescing_analysis TARGET_NAME)
    if(ENABLE_PROFILING)
        if(NOT TARGET ${TARGET_NAME})
            message(WARNING "Target ${TARGET_NAME} does not exist")
            return()
        endif()

        add_custom_target(${TARGET_NAME}_coalescing
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/coalescing_${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions \
                        --export $$PROFILE_DIR/memory_metrics \
                        --print-summary per-kernel \
                        $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/coalescing_analysis.txt 2>&1 && \
                    echo \"Coalescing analysis saved to: $$PROFILE_DIR/coalescing_analysis.txt\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Analyzing memory coalescing for ${TARGET_NAME}"
        )
    endif()
endfunction()

# Function to add occupancy analysis targets
function(add_occupancy_analysis TARGET_NAME)
    if(ENABLE_PROFILING)
        if(NOT TARGET ${TARGET_NAME})
            message(WARNING "Target ${TARGET_NAME} does not exist")
            return()
        endif()

        add_custom_target(${TARGET_NAME}_occupancy
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/occupancy_${TARGET_NAME}_$$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    ncu --metrics achieved_occupancy,sm_occupancy,theoretical_occupancy,eligible_warps_per_cycle \
                        --export $$PROFILE_DIR/occupancy_metrics \
                        --print-summary per-kernel \
                        $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/occupancy_analysis.txt 2>&1 && \
                    echo \"Occupancy analysis saved to: $$PROFILE_DIR/occupancy_analysis.txt\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Analyzing occupancy for ${TARGET_NAME}"
        )
    endif()
endfunction()

# Macro to enable verbose PTX compilation
macro(enable_verbose_ptx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
    message(STATUS "Verbose PTX compilation enabled - will show register usage")
endmacro()

# Function to check if profiling tools are available
function(check_profiling_tools)
    find_program(NSYS_EXECUTABLE nsys)
    find_program(NCU_EXECUTABLE ncu)
    find_program(COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer)

    if(NOT NSYS_EXECUTABLE)
        message(WARNING "nsys not found - Nsight Systems profiling will not be available")
    else()
        message(STATUS "Found Nsight Systems: ${NSYS_EXECUTABLE}")
    endif()

    if(NOT NCU_EXECUTABLE)
        message(WARNING "ncu not found - Nsight Compute profiling will not be available")
    else()
        message(STATUS "Found Nsight Compute: ${NCU_EXECUTABLE}")
    endif()

    if(NOT COMPUTE_SANITIZER_EXECUTABLE)
        message(WARNING "compute-sanitizer not found - CUDA sanitization will not be available")
    else()
        message(STATUS "Found Compute Sanitizer: ${COMPUTE_SANITIZER_EXECUTABLE}")
    endif()
endfunction()

# Print profiling instructions
function(print_profiling_help TARGET_NAME)
    if(ENABLE_PROFILING)
        message(STATUS "")
        message(STATUS "Profiling targets for ${TARGET_NAME}:")
        message(STATUS "  make ${TARGET_NAME}_profile    - Run all profiling tools")
        message(STATUS "  make ${TARGET_NAME}_nsys       - Nsight Systems timeline only")
        message(STATUS "  make ${TARGET_NAME}_ncu        - Nsight Compute kernel analysis only")
        message(STATUS "  make ${TARGET_NAME}_sanitize   - Compute Sanitizer checks only")
        message(STATUS "")
    endif()
endfunction()