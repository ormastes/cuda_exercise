cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA toolkit root
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CUDAToolkit_ROOT /usr/local/cuda-13.0)

# Set CUDA architectures before project
set(CMAKE_CUDA_ARCHITECTURES 75;80;86)

# Enable CUDA language
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Testing support
enable_testing()

# FetchContent for downloading Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include Google Test's CMake utilities
include(CTest)
include(GoogleTest)

# Global settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    # Profile flags: Optimized with debug info
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O2>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O2>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release flags: Optimize for performance
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O3;-use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O3>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
    option(ENABLE_PROFILING "Enable profiling instrumentation" OFF)
endif()

# Additional profiling flags
if(ENABLE_PROFILING)
    function(add_profiling_targets TARGET_NAME)
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
    endfunction()
else()
    function(add_profiling_targets TARGET_NAME)
        message(WARNING "ENABLE_PROFILING variable is not enabled")
    endfunction()    
endif()

# Verbose PTX compilation (shows register usage)
option(VERBOSE_PTXAS "Show PTX compilation details" OFF)
if(VERBOSE_PTXAS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()


add_subdirectory(00.demo)
add_subdirectory(00.demo_lib)
add_subdirectory(10.cuda_basic)