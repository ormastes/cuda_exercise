project(50_GPU_NVMe_Interaction)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Check for GPUDirect Storage (optional but recommended)
find_library(CUFILE_LIB cufile PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
if(CUFILE_LIB)
    message(STATUS "Found cuFile library for GPUDirect Storage: ${CUFILE_LIB}")
    set(HAS_GDS ON)
else()
    message(WARNING "cuFile library not found. GPUDirect Storage features will be disabled.")
    set(HAS_GDS OFF)
endif()

# Check for SPDK (optional)
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(SPDK spdk_nvme)
endif()

# Source files
set(NVME_SOURCES)

if(HAS_GDS)
    list(APPEND NVME_SOURCES
        gds_nvme_io.cu
    )
endif()

if(SPDK_FOUND)
    list(APPEND NVME_SOURCES
        spdk_nvme_gpu.cu
    )
endif()

# Always include the high-level API
list(APPEND NVME_SOURCES
    nvme_api.cu
    nvme_python_wrapper.cu
)

# Create library only if we have sources
if(NVME_SOURCES)
    add_library(${PROJECT_NAME}_lib ${NVME_SOURCES})

    target_include_directories(${PROJECT_NAME}_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(${PROJECT_NAME}_lib PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        ${CUDA_BASIC_LIB}
        Threads::Threads
    )

    if(HAS_GDS)
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUFILE_LIB})
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HAS_GDS=1)
    endif()

    if(SPDK_FOUND)
        target_include_directories(${PROJECT_NAME}_lib PUBLIC ${SPDK_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${SPDK_LIBRARIES})
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HAS_SPDK=1)
    endif()

    # Example executables
    if(HAS_GDS)
        add_executable(${PROJECT_NAME}_demo_gds
            gds_nvme_io.cu
        )
        target_link_libraries(${PROJECT_NAME}_demo_gds
            ${PROJECT_NAME}_lib
            ${CUDA_BASIC_LIB}
        )
    endif()

    if(SPDK_FOUND)
        add_executable(${PROJECT_NAME}_demo_spdk
            spdk_nvme_gpu.cu
        )
        target_link_libraries(${PROJECT_NAME}_demo_spdk
            ${PROJECT_NAME}_lib
            ${CUDA_BASIC_LIB}
        )
    endif()

    # Tests
    if(BUILD_TESTING)
        set(TEST_SOURCES)

        if(HAS_GDS)
            list(APPEND TEST_SOURCES test_gds_nvme_io.cu)
        endif()

        if(SPDK_FOUND)
            list(APPEND TEST_SOURCES test_spdk_nvme_gpu.cu)
        endif()

        list(APPEND TEST_SOURCES
            test_nvme_api.cu
            test_python_wrapper.cu
        )

        if(TEST_SOURCES)
            add_executable(${PROJECT_NAME}_test ${TEST_SOURCES})
            target_link_libraries(${PROJECT_NAME}_test
                ${PROJECT_NAME}_lib
                ${GTEST_BASIC_LIB}
                GTest::gtest_main
                GTestCudaGenerator
            )
            gtest_discover_tests(${PROJECT_NAME}_test)
        endif()
    endif()

    # Profiling targets
    if(HAS_GDS)
        add_profiling_targets(${PROJECT_NAME}_demo_gds)
    endif()
    if(SPDK_FOUND)
        add_profiling_targets(${PROJECT_NAME}_demo_spdk)
    endif()

    # Python bindings (optional)
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        # PyCUDA wrapper
        add_library(pycuda_nvme SHARED
            nvme_python_wrapper.cu
        )
        target_link_libraries(pycuda_nvme
            ${PROJECT_NAME}_lib
            Python3::Python
        )
        set_target_properties(pycuda_nvme PROPERTIES
            PREFIX ""
            SUFFIX ".so"
        )

        # PyTorch extension (if PyTorch is available)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])"
            OUTPUT_VARIABLE TORCH_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(TORCH_INCLUDE_DIR)
            add_library(torch_nvme SHARED
                nvme_python_wrapper.cu
            )
            target_include_directories(torch_nvme PRIVATE ${TORCH_INCLUDE_DIR})
            target_link_libraries(torch_nvme
                ${PROJECT_NAME}_lib
                ${Python3_LIBRARIES}
            )
            target_compile_definitions(torch_nvme PRIVATE USE_PYTORCH=1)
            set_target_properties(torch_nvme PROPERTIES
                PREFIX ""
                SUFFIX ".so"
            )
        endif()
    endif()

else()
    message(WARNING "No NVMe sources available. Install GPUDirect Storage or SPDK for functionality.")
endif()