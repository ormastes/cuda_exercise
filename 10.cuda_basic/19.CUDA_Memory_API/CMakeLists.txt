project(19_CUDA_Memory_API)

# Main executable for memory API demonstrations
add_executable(${PROJECT_NAME} memory_api_demo.cu)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Enable device-side debug symbols in debug mode
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
)

# Create library for testing (exports kernels)
add_library(${PROJECT_NAME}_lib STATIC memory_api_demo.cu)

target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    BUILDING_LIB  # Define to avoid main() in library
)

# Unit tests
add_executable(${PROJECT_NAME}_test test_memory_api.cu)

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

# Custom target for memory transfer analysis
add_custom_target(${PROJECT_NAME}_transfer_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Transfer Performance ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/transfer_analysis_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Running memory transfer benchmark...' && \
            $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/output.txt && \
            echo 'Profiling memory transfers...' && \
            (nsys profile --stats=true \
                 --output=$$PROFILE_DIR/memory_transfers \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/nsys_output.txt 2>&1 || echo 'NSYS profiling complete') && \
            echo 'Transfer analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory transfer patterns and bandwidth"
)

# Custom target for unified memory analysis
add_custom_target(${PROJECT_NAME}_unified_memory_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Unified Memory Behavior ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/unified_memory_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Profiling unified memory migrations...' && \
            (ncu --metrics gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_bytes.sum \
                 --kernel-name unified_memory_kernel \
                 --export $$PROFILE_DIR/unified_metrics \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/ncu_output.txt 2>&1 || echo 'NCU analysis complete') && \
            echo 'Unified memory analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing unified memory page migrations"
)

# Custom target for pinned memory comparison
add_custom_target(${PROJECT_NAME}_pinned_memory_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Comparing Pinned vs Pageable Memory ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/pinned_memory_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Analyzing pinned memory performance...' && \
            $<TARGET_FILE:${PROJECT_NAME}> | grep -A 5 'Pinned Memory' > $$PROFILE_DIR/pinned_comparison.txt && \
            (nsys profile --stats=true \
                 --output=$$PROFILE_DIR/pinned_profile \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/nsys_pinned.txt 2>&1 || true) && \
            echo 'Pinned memory analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Comparing pinned and pageable memory transfer performance"
)

# Custom target for memory pool analysis (CUDA 11.2+)
add_custom_target(${PROJECT_NAME}_memory_pool_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Memory Pool Operations ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/memory_pools_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Testing memory pool support...' && \
            $<TARGET_FILE:${PROJECT_NAME}> | grep -A 5 'Memory Pools' > $$PROFILE_DIR/pool_output.txt && \
            echo 'Memory pool analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing memory pool operations (CUDA 11.2+ required)"
)

# Custom target for async operations analysis
add_custom_target(${PROJECT_NAME}_async_analysis
    COMMAND ${CMAKE_COMMAND} -E echo "=== Analyzing Asynchronous Memory Operations ==="
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
    COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/async_ops_$$(date +%Y%m%d_%H%M%S) && \
            mkdir -p $$PROFILE_DIR && \
            echo 'Profiling async operations...' && \
            (nsys profile --stats=true \
                 --output=$$PROFILE_DIR/async_profile \
                 --trace=cuda,nvtx \
                 $<TARGET_FILE:${PROJECT_NAME}> > $$PROFILE_DIR/nsys_async.txt 2>&1 || true) && \
            echo 'Async operations analysis saved to: $$PROFILE_DIR'"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Analyzing asynchronous memory operations with multiple streams"
)

# Add test for the main executable
if(BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_main_test
        COMMAND ${PROJECT_NAME}
    )
    set_tests_properties(${PROJECT_NAME}_main_test PROPERTIES
        PASS_REGULAR_EXPRESSION "CUDA Memory API Demo Complete"
        TIMEOUT 30
    )
endif()