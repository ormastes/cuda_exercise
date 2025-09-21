project(intermediate_cuda)

# Placeholder for intermediate CUDA examples
# Each part will have its own subdirectory when implemented

# Part 10: Synchronization and Atomics
# add_subdirectory("10.Synchronization and Atomics")

# Part 11: Streams and Asynchronous Execution
# add_subdirectory("11.Streams and Async")

# Part 12: Unified Memory
# add_subdirectory("12.Unified Memory")

# Part 13: Pinned Memory and Zero-Copy
# add_subdirectory("13.Pinned Memory")

# Part 14: GPUDirect and Storage
# add_subdirectory("14.GPUDirect Storage")

# Part 15: CUDA Libraries
# add_subdirectory("15.CUDA Libraries")

# Part 16: Multi-GPU Programming
# add_subdirectory("16.Multi-GPU")

# Part 17: Performance Optimization
# add_subdirectory("17.Performance Optimization")

# Part 18: cuDNN (Optional)
# add_subdirectory("18.cuDNN")

# For now, create a simple info target
add_custom_target(intermediate_cuda_info
    COMMAND ${CMAKE_COMMAND} -E echo "=== Intermediate CUDA Programming ==="
    COMMAND ${CMAKE_COMMAND} -E echo "This section contains advanced CUDA topics."
    COMMAND ${CMAKE_COMMAND} -E echo "Parts 10-18 are planned for implementation."
    COMMAND ${CMAKE_COMMAND} -E echo "See 20.intermediate_cuda/README.md for details."
    COMMENT "Information about intermediate CUDA section"
)