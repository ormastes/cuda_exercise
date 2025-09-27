# 💾 Part 7: Memory Hierarchy

**Goal**: Master CUDA memory hierarchy through matrix multiplication optimization, demonstrating the performance impact of different memory access patterns.

---

## **7.1 Overview**

The CUDA memory hierarchy is crucial for achieving high performance. This section demonstrates:
- Global memory access patterns and coalescing
- Shared memory usage and tiling techniques
- Bank conflict mitigation
- Cache optimization strategies

---

## **7.2 Memory Types and Characteristics**

| Memory Type | Scope | Bandwidth | Latency | Size | Cached |
|-------------|-------|-----------|---------|------|--------|
| **Registers** | Thread | Highest | Lowest | ~255 per thread | No |
| **Shared** | Block | Very High | Very Low | 48-96 KB/block | No |
| **L1 Cache** | SM | High | Low | 128 KB/SM | Yes |
| **L2 Cache** | Device | High | Medium | 4-6 MB | Yes |
| **Global** | Device | Medium | High | 4-48 GB | Through L1/L2 |
| **Constant** | Device | High | Low | 64 KB | Yes |
| **Texture** | Device | High | Medium | Via cache | Yes |

---

## **7.3 Implementation Strategies**

### **7.3.1 Naive Implementation**
```cpp
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```
**Issues**: Poor cache utilization, uncoalesced memory access for matrix B

### **7.3.2 Coalesced Memory Access**
- **Coalesced**: Consecutive threads access consecutive memory addresses
- **Strided**: Threads access memory with gaps
- **Random**: No pattern in memory access

### **7.3.3 Shared Memory Tiling**
```cpp
#define TILE_SIZE 16

__global__ void matmul_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Load tiles into shared memory
    // Compute partial products
    // Accumulate results
}
```
**Benefits**: Reduces global memory access by factor of TILE_SIZE

### **7.3.4 Bank Conflict Mitigation**
```cpp
// With padding to avoid bank conflicts
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];
```
**Purpose**: Padding shifts memory addresses to different banks

---

## **7.4 Performance Optimization Techniques**

### **Memory Coalescing Rules**
1. **Aligned Access**: Start addresses should be aligned to 128 bytes
2. **Sequential Access**: Threads should access consecutive addresses
3. **Size Matching**: Access size should match word size (32/64/128 bits)

### **Shared Memory Best Practices**
1. **Minimize Bank Conflicts**: Use padding or access patterns that spread across banks
2. **Double Buffering**: Overlap computation with data loading
3. **Reuse Data**: Load once, use multiple times

### **Cache Optimization**
1. **Spatial Locality**: Access nearby data together
2. **Temporal Locality**: Reuse recently accessed data
3. **Cache Blocking**: Structure algorithms to fit in cache

---

## **7.5 Running the Examples**

### **Building**
```bash
cd build
cmake --build . --target 17_Memory_Hierarchy
```

### **Running Main Demo**
```bash
# Run with default matrix size (512x512)
./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy

# Run with custom size
./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy 1024
```

### **Running Tests**
```bash
# Run all tests
ctest -R 17_Memory_Hierarchy

# Run with verbose output
./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_test
```

---

## **7.6 Profiling and Analysis**

### **Memory Pattern Analysis**
```bash
# Analyze memory access patterns
make 17_Memory_Hierarchy_memory_analysis
```

### **Bank Conflict Analysis**
```bash
# Check for shared memory bank conflicts
make 17_Memory_Hierarchy_bank_conflict_analysis
```

### **Performance Comparison**
```bash
# Compare performance across matrix sizes
make 17_Memory_Hierarchy_performance_comparison
```

### **Cache Analysis**
```bash
# Analyze L1/L2 cache performance
make 17_Memory_Hierarchy_cache_analysis
```

---

## **7.7 Expected Output**

```
Using device: NVIDIA TITAN RTX
Compute capability: 7.5
Shared memory per block: 48 KB
Max threads per block: 1024

=== Memory Access Pattern Demonstration ===
Strided access time: 0.234 ms
Coalesced access time: 0.045 ms
Speedup from coalescing: 5.2x

=== Memory Hierarchy Benchmark ===
Matrix size: 512 x 512

Kernel Performance:
------------------------------------------------------------
    Naive (poor memory access):    15.234 ms,   35.12 GFLOPS
    Coalesced (better access pattern):    12.456 ms,   42.98 GFLOPS
    Shared Memory (tiled):     4.123 ms,  129.87 GFLOPS
    Shared Memory (bank-conflict free):     3.892 ms,  137.54 GFLOPS

Verifying correctness...
Results verified: PASS

=== Memory Hierarchy Demo Complete ===
```

---

## **7.8 Performance Analysis**

### **Speedup Factors**
| Implementation | Relative Speedup | Key Optimization |
|----------------|------------------|------------------|
| Naive | 1.0x (baseline) | None |
| Coalesced | 1.2-1.5x | Better memory access pattern |
| Shared Memory | 3.5-4.0x | Reduced global memory access |
| Bank-Conflict Free | 3.8-4.2x | Eliminated shared memory conflicts |

### **Memory Bandwidth Utilization**
```
Naive:           ~30% of theoretical peak
Coalesced:       ~40% of theoretical peak
Shared Memory:   ~75% of theoretical peak
Optimized:       ~80% of theoretical peak
```

---

## **7.9 Common Pitfalls and Solutions**

### **Problem 1: Uncoalesced Memory Access**
**Symptom**: Poor performance despite parallel execution
**Solution**: Restructure data layout or access pattern

### **Problem 2: Shared Memory Bank Conflicts**
**Symptom**: Slower than expected shared memory performance
**Solution**: Add padding or reorganize access pattern

### **Problem 3: Register Spilling**
**Symptom**: Local memory usage shown in profiler
**Solution**: Reduce per-thread register usage

### **Problem 4: Low Occupancy**
**Symptom**: Low SM utilization
**Solution**: Adjust block size or reduce resource usage

---

## **7.10 Exercises**

### **Exercise 1: Different Tile Sizes**
Modify TILE_SIZE and measure performance impact:
```cpp
#define TILE_SIZE 8   // Try: 8, 16, 32
```

### **Exercise 2: Double Buffering**
Implement double buffering for shared memory to overlap loads with computation.

### **Exercise 3: Rectangular Tiles**
Experiment with non-square tiles (e.g., 16x32) for better memory access.

### **Exercise 4: Multi-level Tiling**
Implement tiling at both L1 and L2 cache levels.

---

## **7.11 Profiler Metrics to Monitor**

### **Key Metrics**
- `gld_efficiency`: Global load efficiency (target: >80%)
- `gst_efficiency`: Global store efficiency (target: >80%)
- `shared_efficiency`: Shared memory efficiency (target: >90%)
- `l1_cache_global_hit_rate`: L1 cache hit rate
- `l2_cache_hit_rate`: L2 cache hit rate
- `achieved_occupancy`: Actual vs theoretical occupancy

### **Using Nsight Compute**
```bash
ncu --metrics gld_efficiency,shared_efficiency ./17_Memory_Hierarchy 256
```

---

## **7.12 Advanced Topics**

### **Texture Memory**
- Optimized for 2D spatial locality
- Hardware interpolation support
- Cached through separate texture cache

### **Constant Memory**
- Broadcast reads to all threads
- Cached through constant cache
- Limited to 64KB total

### **Unified Memory**
- Automatic data migration
- Simplified programming model
- Performance overhead for migration

---

## **7.13 Best Practices Summary**

1. **Always profile before optimizing** - Identify actual bottlenecks
2. **Coalesce memory access** - Critical for global memory performance
3. **Use shared memory effectively** - Trade off capacity vs parallelism
4. **Minimize data movement** - Reuse data when possible
5. **Consider memory hierarchy** - Optimize for each level
6. **Monitor occupancy** - Balance resource usage
7. **Avoid bank conflicts** - Use padding or access patterns

---

## **7.14 References**

- [CUDA Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [Nsight Compute User Manual](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Matrix Multiplication Optimization](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda/)

---

## ✅ **Summary**

This section demonstrated:
- Impact of memory access patterns on performance
- Shared memory tiling for bandwidth reduction
- Bank conflict mitigation strategies
- Cache optimization techniques
- Profiling and analysis methods

**Key Takeaways:**
- Memory optimization can yield 4-5x speedups
- Coalescing is essential for global memory performance
- Shared memory reduces bandwidth requirements
- Profiling guides optimization efforts

---

📄 **Next**: Part 8 - Thread Hierarchy Practice