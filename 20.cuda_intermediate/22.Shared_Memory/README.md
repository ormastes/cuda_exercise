# Shared Memory in CUDA

## Introduction

Shared memory is a crucial optimization technique in CUDA programming that provides fast, on-chip memory shared among threads within a block. It acts as a user-managed cache that is significantly faster than global memory.

## Key Concepts

### Memory Hierarchy
- **Registers**: Fastest, thread-private
- **Shared Memory**: Fast, block-shared (this tutorial)
- **L1/L2 Cache**: Automatic caching
- **Global Memory**: Slowest, accessible by all threads

### Shared Memory Characteristics
- Located on-chip (same silicon as GPU cores)
- ~100x faster than global memory
- Limited size (48KB-164KB per SM depending on architecture)
- Scope: Thread block (threads in same block can share data)
- Lifetime: Block execution

## Benefits

1. **Reduced Memory Latency**: On-chip location provides low-latency access
2. **Memory Bandwidth Reduction**: Reuse data within thread blocks
3. **Inter-thread Communication**: Efficient data sharing between threads
4. **Coalescing Optimization**: Stage uncoalesced global memory accesses

## Common Use Cases

### 1. Matrix Multiplication (Tiling)
Load tiles into shared memory to reuse data multiple times

### 2. Reduction Operations
Efficiently share partial results among threads

### 3. Stencil Operations
Cache halo regions for neighbor access patterns

### 4. Data Transposition
Use shared memory to avoid strided global memory access

## Declaration and Allocation

### Static Allocation
```cuda
__shared__ float sharedArray[256];
```

### Dynamic Allocation
```cuda
extern __shared__ float dynamicShared[];
// Launch with: kernel<<<blocks, threads, sharedMemSize>>>();
```

### Bank-Conflict-Free Access
Shared memory is divided into banks (32 banks on modern GPUs). Simultaneous access to different addresses in the same bank causes serialization.

## Examples in This Section

1. **basic_shared_memory.cu**: Introduction to shared memory usage
2. **matrix_transpose.cu**: Efficient matrix transposition using shared memory
3. **reduction.cu**: Sum reduction with shared memory optimization
4. **stencil_1d.cu**: 1D stencil computation with shared memory caching

## Performance Considerations

### Occupancy Impact
- Shared memory usage affects occupancy
- Balance between shared memory per block and active blocks per SM

### Bank Conflicts
- Stride-1 access pattern is optimal
- Use padding to avoid conflicts in 2D arrays

### Synchronization Overhead
- `__syncthreads()` is necessary but has overhead
- Minimize synchronization points

## Best Practices

1. **Coalesce Global Memory Access**: Use shared memory to stage uncoalesced accesses
2. **Maximize Reuse**: Load once, use multiple times
3. **Consider Cache**: L1 cache may be sufficient for simple access patterns
4. **Profile Performance**: Use Nsight Compute to analyze shared memory efficiency
5. **Handle Edge Cases**: Properly handle boundaries when tile size doesn't divide evenly

## Architecture-Specific Limits

| Architecture | Compute Capability | Max Shared Memory per Block | Max Shared Memory per SM |
|--------------|-------------------|----------------------------|-------------------------|
| Pascal       | 6.x               | 48 KB                      | 64 KB                   |
| Volta/Turing | 7.x               | 96 KB (configurable)       | 96 KB                   |
| Ampere       | 8.x               | 164 KB (configurable)      | 164 KB                  |
| Ada Lovelace | 8.9               | 164 KB (configurable)      | 164 KB                  |
| Hopper       | 9.0               | 228 KB (configurable)      | 228 KB                  |

## Compile and Run

```bash
# Build all examples
mkdir build && cd build
cmake ..
make

# Run individual examples
./basic_shared_memory
./matrix_transpose
./reduction
./stencil_1d

# Profile with Nsight Compute
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./matrix_transpose
```

## Further Reading

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [CUDA C++ Best Practices Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)