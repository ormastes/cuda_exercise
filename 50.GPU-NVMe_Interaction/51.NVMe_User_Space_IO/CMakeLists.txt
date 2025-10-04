project(51_GPU_Optimizations)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Check for cuBLAS and cuDNN for comparisons
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# Optional: Find cuDNN for advanced optimizations
find_library(CUDNN_LIB cudnn PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
if(CUDNN_LIB)
    message(STATUS "Found cuDNN library: ${CUDNN_LIB}")
    set(HAS_CUDNN ON)
else()
    message(STATUS "cuDNN not found. Some optimizations will be disabled.")
    set(HAS_CUDNN OFF)
endif()

# Source files - Core optimization implementations
set(OPTIMIZATION_SOURCES
    src/kernels/cpu_baseline.cu
    src/kernels/matmul_optimized.cu
    src/kernels/backprop_optimized.cu
    src/kernels/attention_optimized.cu
    src/kernels/moe_optimized.cu
    src/kernels/pytorch_bindings.cu
)

# Check if Part 50 (GPU-NVMe) is available as a sibling module
if(EXISTS "${CMAKE_SOURCE_DIR}/50.GPU-NVMe_Interaction/CMakeLists.txt")
    set(HAS_NVME_SUPPORT ON)
    list(APPEND OPTIMIZATION_SOURCES src/kernels/nvme_data_loader.cu)
    message(STATUS "Found Part 50 (GPU-NVMe Interaction). NVMe data loading enabled.")
else()
    set(HAS_NVME_SUPPORT OFF)
    message(STATUS "Part 50 not found. NVMe data loading will be disabled.")
endif()

# Create main library
add_library(${PROJECT_NAME}_lib ${OPTIMIZATION_SOURCES})

target_include_directories(${PROJECT_NAME}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    $<$<BOOL:${HAS_NVME_SUPPORT}>:${CMAKE_SOURCE_DIR}/50.GPU-NVMe_Interaction>
)

target_link_libraries(${PROJECT_NAME}_lib PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    ${CUDA_BASIC_LIB}
    Threads::Threads
)

# Compile options for optimizations
target_compile_options(${PROJECT_NAME}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extra-device-vectorization
        -arch=sm_70
    >
)

# Enable Tensor Core operations if available (SM 7.0+)
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    USE_TENSOR_CORES=1
    $<$<BOOL:${HAS_CUDNN}>:HAS_CUDNN=1>
    $<$<BOOL:${HAS_NVME_SUPPORT}>:HAS_NVME_SUPPORT=1>
)

if(HAS_CUDNN)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDNN_LIB})
endif()

if(HAS_NVME_SUPPORT)
    # Link to parent NVMe library if available
    if(TARGET 50_GPU_NVMe_Interaction_lib)
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC 50_GPU_NVMe_Interaction_lib)
    else()
        message(STATUS "NVMe library target not found. Building without direct NVMe linking.")
    endif()
endif()

# Benchmark executable
add_executable(${PROJECT_NAME}_benchmark
    src/kernels/cpu_baseline.cu
    src/kernels/matmul_optimized.cu
    src/kernels/backprop_optimized.cu
    src/kernels/attention_optimized.cu
    src/kernels/moe_optimized.cu
)

target_link_libraries(${PROJECT_NAME}_benchmark
    ${PROJECT_NAME}_lib
    ${CUDA_BASIC_LIB}
)

# Individual optimization demos
add_executable(${PROJECT_NAME}_matmul
    src/kernels/matmul_optimized.cu
)
target_link_libraries(${PROJECT_NAME}_matmul
    ${PROJECT_NAME}_lib
    ${CUDA_BASIC_LIB}
)

add_executable(${PROJECT_NAME}_attention
    src/kernels/attention_optimized.cu
)
target_link_libraries(${PROJECT_NAME}_attention
    ${PROJECT_NAME}_lib
    ${CUDA_BASIC_LIB}
)

# Tests
if(BUILD_TESTING)
    set(TEST_SOURCES
        test/kernels/test_cpu_baseline.cu
        test/kernels/test_matmul_optimized.cu
        test/kernels/test_backprop_optimized.cu
        test/kernels/test_attention_optimized.cu
        test/kernels/test_moe_optimized.cu
        test/kernels/test_pytorch_bindings.cu
    )

    add_executable(${PROJECT_NAME}_test ${TEST_SOURCES})
    target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}_lib
        ${GTEST_BASIC_LIB}
        GTest::gtest_main
        GTestCudaGenerator
        CUDA::cublas
    )
    gtest_discover_tests(${PROJECT_NAME}_test)
endif()

# Python bindings
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    # PyCUDA wrapper for CPU baseline
    add_library(pycuda_cpu_ops SHARED
        src/kernels/cpu_baseline.cu
    )
    target_link_libraries(pycuda_cpu_ops
        ${PROJECT_NAME}_lib
        Python3::Python
    )
    set_target_properties(pycuda_cpu_ops PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )

    # PyCUDA wrapper for optimized operations
    add_library(pycuda_optimized_ops SHARED
        src/kernels/pytorch_bindings.cu
    )
    target_link_libraries(pycuda_optimized_ops
        ${PROJECT_NAME}_lib
        Python3::Python
    )
    set_target_properties(pycuda_optimized_ops PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )

    # PyTorch extension
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
        OUTPUT_VARIABLE TORCH_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(TORCH_INCLUDE_DIR)
        message(STATUS "Found PyTorch ${TORCH_VERSION}: ${TORCH_INCLUDE_DIR}")

        # Find Torch libraries
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
            OUTPUT_VARIABLE TORCH_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # PyTorch C++ extension
        add_library(optimized_ops SHARED
            src/kernels/pytorch_bindings.cu
        )

        target_include_directories(optimized_ops PRIVATE
            ${TORCH_INCLUDE_DIR}
            ${TORCH_INCLUDE_DIR}/torch/csrc/api/include
        )

        target_link_libraries(optimized_ops
            ${PROJECT_NAME}_lib
            ${Python3_LIBRARIES}
            "${TORCH_PATH}/lib/libtorch.so"
            "${TORCH_PATH}/lib/libc10.so"
            "${TORCH_PATH}/lib/libc10_cuda.so"
        )

        target_compile_definitions(optimized_ops PRIVATE
            USE_PYTORCH=1
            TORCH_EXTENSION_NAME=optimized_ops
        )

        set_target_properties(optimized_ops PROPERTIES
            PREFIX ""
            SUFFIX ".so"
            CXX_STANDARD 17
        )

        # Install Python module
        install(TARGETS optimized_ops pycuda_optimized_ops pycuda_cpu_ops
                LIBRARY DESTINATION ${Python3_SITELIB})
    else()
        message(STATUS "PyTorch not found. PyTorch extensions will not be built.")
    endif()
endif()

# Profiling targets
add_profiling_targets(${PROJECT_NAME}_benchmark)
add_profiling_targets(${PROJECT_NAME}_matmul)
add_profiling_targets(${PROJECT_NAME}_attention)

# Examples directory
add_custom_target(examples
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/examples
            ${CMAKE_CURRENT_BINARY_DIR}/examples
    COMMENT "Copying example scripts"
)

# Performance reporting
add_custom_target(benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_benchmark
    DEPENDS ${PROJECT_NAME}_benchmark
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running performance benchmarks"
)