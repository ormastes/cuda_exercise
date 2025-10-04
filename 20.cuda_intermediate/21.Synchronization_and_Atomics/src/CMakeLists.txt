# Source library for synchronization and atomics
add_library(${MODULE}_lib STATIC
    kernels/atomic_operations.cu
    kernels/synchronization_primitives.cu
    kernels/lock_free_structures.cu
)

target_compile_definitions(${MODULE}_lib PUBLIC BUILDING_LIB)

target_compile_options(${MODULE}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
)

set_target_properties(${MODULE}_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${MODULE}_lib PUBLIC CudaCustomLib)

# Example executables
add_executable(${MODULE}_histogram examples/histogram_atomic.cu)
target_link_libraries(${MODULE}_histogram PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})

add_executable(${MODULE}_producer_consumer examples/producer_consumer.cu)
target_link_libraries(${MODULE}_producer_consumer PRIVATE ${MODULE}_lib ${CUDA_BASIC_LIB})
