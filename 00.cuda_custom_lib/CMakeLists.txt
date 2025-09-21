# CMakeLists.txt for CudaCustomLib - Header-only CUDA utilities library

# Create an interface library (header-only)
add_library(CudaCustomLib INTERFACE)

# Add include directories for the interface
target_include_directories(CudaCustomLib
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link CUDA runtime (users of this library will automatically get CUDA runtime)
target_link_libraries(CudaCustomLib
    INTERFACE
    CUDA::cudart
)

# Set required C++ standard
target_compile_features(CudaCustomLib
    INTERFACE
    cxx_std_17
)

# Optional: Add compile definitions for debug builds
target_compile_definitions(CudaCustomLib
    INTERFACE
    $<$<CONFIG:Debug>:DEBUG>
)

# Create an alias for cleaner usage
add_library(Cuda::Utils ALIAS CudaCustomLib)