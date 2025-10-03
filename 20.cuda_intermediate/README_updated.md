# 🚀 Intermediate CUDA Programming

**Goal**: Master advanced CUDA features including synchronization, memory optimization, multi-GPU programming, and dynamic parallelism.

## Module Navigation

### Synchronization & Concurrency
- [21. Synchronization and Atomics](21.Synchronization_and_Atomics/README.md) - Thread synchronization and atomic operations
- [22. Streams and Async](22.Streams_and_Async/README.md) - Concurrent kernel execution and async operations

### Memory Optimization
- [23. Shared Memory](23.Shared_Memory/README.md) - Shared memory usage and optimization
- [24. Memory Coalescing and Bank Conflicts](24.Memory_Coalescing_and_Bank_Conflicts/README.md) - Memory access patterns

### Advanced Features
- [25. Dynamic Parallelism](25.Dynamic_Parallelism/README.md) - Launching kernels from device code
- [26. Cooperative Groups Advanced](26.Cooperative_Groups_Advanced/README.md) - Advanced thread cooperation
- [27. Multi-GPU Programming](27.Multi_GPU_Programming/README.md) - Scaling across multiple GPUs

---

## Prerequisites

Before starting this section, you should have completed:
- All modules from [10.cuda_basic](../10.cuda_basic/README.md)
- Understanding of CUDA memory hierarchy
- Proficiency with kernel debugging and profiling

## Learning Path

### Phase 1: Concurrency (Modules 21-22)
Start with synchronization primitives and atomic operations, then move to streams for overlapping computation and transfers.

### Phase 2: Memory Optimization (Modules 23-24)
Master shared memory usage and understand memory coalescing for maximum bandwidth utilization.

### Phase 3: Advanced Features (Modules 25-27)
Explore dynamic parallelism for recursive algorithms, cooperative groups for flexible synchronization, and multi-GPU programming for large-scale applications.

## Key Concepts Covered

1. **Synchronization Primitives**
   - Barriers and fences
   - Atomic operations
   - Memory consistency

2. **Concurrent Execution**
   - CUDA streams
   - Event management
   - Kernel concurrency

3. **Memory Optimization**
   - Shared memory tiling
   - Bank conflict avoidance
   - Coalesced access patterns

4. **Advanced Programming Models**
   - Dynamic parallelism
   - Cooperative groups
   - Multi-GPU coordination

## Performance Expectations

After completing this section, you should achieve:
- 80%+ memory bandwidth efficiency
- Effective stream overlap for computation/transfer
- Linear scaling with multiple GPUs (70%+ efficiency)

## Common Challenges

| Challenge | Solution | Module |
|-----------|----------|--------|
| Race conditions | Proper synchronization | 21 |
| Low bandwidth | Coalesced access | 24 |
| GPU underutilization | Stream concurrency | 22 |
| Limited memory | Shared memory tiling | 23 |
| Single GPU bottleneck | Multi-GPU scaling | 27 |

## Building and Testing

Each module includes:
- Source code in `src/` directory
- Unit tests in `test/` directory
- Performance benchmarks
- CMakeLists.txt for building

```bash
# Build all intermediate modules
cmake -B build
cmake --build build

# Run tests for specific module
ctest -R "21_*"  # Test module 21
```

## Next Steps

After completing intermediate topics:
- 📚 Continue to [30.CUDA_Libraries](../30.CUDA_Libraries/README.md) for library usage
- 🚀 Or jump to [40.cuda_advanced](../40.cuda_advanced/README.md) for cutting-edge features

---

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)