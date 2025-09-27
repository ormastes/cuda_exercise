# 🔀 Part 8: Thread Hierarchy Practice

**Goal**: Master CUDA thread hierarchy through advanced tiling strategies, warp optimization, and occupancy tuning.

---

## **8.1 Overview**

The CUDA thread hierarchy is fundamental to GPU performance. This section explores:
- Thread organization (threads, warps, blocks, grids)
- Tiling strategies and tile size selection
- Warp-level optimizations
- Occupancy analysis and tuning
- Thread coarsening techniques
- Avoiding warp divergence

---

## **8.2 Thread Hierarchy Components**

### **Hierarchy Levels**

| Level | Size | Scope | Synchronization | Memory Access |
|-------|------|-------|------------------|---------------|
| **Thread** | 1 | Individual | N/A | Registers, Local |
| **Warp** | 32 threads | SIMT unit | Implicit | Shared via shuffle |
| **Block** | ≤1024 threads | Cooperative group | `__syncthreads()` | Shared memory |
| **Grid** | All blocks | Kernel launch | Kernel completion | Global memory |

### **Hardware Limits (Typical)**
```
Max threads per block: 1024
Max thread dimensions: (1024, 1024, 64)
Max grid dimensions: (2³¹-1, 65535, 65535)
Warp size: 32 (fixed)
Max warps per SM: 64
Max blocks per SM: 32
```

---

## **8.3 Tiling Strategies**

### **8.3.1 Square Tiles**
```cpp
template<int TILE_SIZE>
__global__ void matmul_tiled_basic(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    // Load tiles cooperatively
    // Compute tile products
    // Accumulate results
}
```

**Tile Size Selection:**
- **8x8**: 64 threads/block - Low occupancy, good for small problems
- **16x16**: 256 threads/block - Balanced choice
- **32x32**: 1024 threads/block - Maximum threads, may limit occupancy

### **8.3.2 Rectangular Tiles**
```cpp
template<int TILE_Y, int TILE_X>
__global__ void matmul_rectangular_tiles(...) {
    __shared__ float As[TILE_Y][TILE_X];
    __shared__ float Bs[TILE_Y][TILE_X];
    // Better for non-square matrices
    // Can improve memory access patterns
}
```

**Benefits:**
- Better thread utilization for non-square problems
- Can optimize for specific memory access patterns
- Flexible work distribution

### **8.3.3 Thread Coarsening**
```cpp
template<int COARSE_FACTOR>
__global__ void matmul_thread_coarsening(...) {
    float sum[COARSE_FACTOR];  // Each thread computes multiple elements
    // Reduces thread count
    // Increases work per thread
    // Can improve instruction-level parallelism
}
```

---

## **8.4 Warp-Level Optimization**

### **Warp Execution Model**
- 32 threads execute in lockstep (SIMT)
- Same instruction for all threads in warp
- Divergence causes serialization

### **Warp Shuffle Instructions**
```cpp
// Warp-level primitives (compute capability 3.0+)
__shfl_sync(mask, value, srcLane)     // Read from specific lane
__shfl_up_sync(mask, value, delta)    // Read from lane ID - delta
__shfl_down_sync(mask, value, delta)  // Read from lane ID + delta
__shfl_xor_sync(mask, value, laneMask) // XOR lane ID
```

### **Avoiding Warp Divergence**
```cpp
// BAD: Divergence within warp
if (threadIdx.x % 2 == 0) {
    // Even threads
} else {
    // Odd threads - DIVERGENT!
}

// GOOD: No divergence within warp
if ((threadIdx.x / 32) % 2 == 0) {
    // Even warps
} else {
    // Odd warps - no divergence within warp
}
```

---

## **8.5 Occupancy Analysis**

### **Occupancy Factors**

| Resource | Limit | Impact on Occupancy |
|----------|-------|---------------------|
| **Registers** | 65536 per SM | `Occupancy = min(1, MaxRegs / (RegsPerThread × ThreadsPerBlock))` |
| **Shared Memory** | 48-96 KB per SM | `Occupancy = min(1, MaxShared / SharedPerBlock)` |
| **Threads** | 2048 per SM | `Occupancy = min(1, MaxThreads / ThreadsPerBlock)` |
| **Blocks** | 32 per SM | `Occupancy = min(1, MaxBlocks / BlocksNeeded)` |

### **Occupancy Calculator API**
```cpp
int min_grid_size, optimal_block_size;
cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size,
    &optimal_block_size,
    kernel_function,
    dynamic_shared_mem_size,
    max_block_size);
```

---

## **8.6 Running the Examples**

### **Building**
```bash
cd build
cmake --build . --target 18_Thread_Hierarchy_Practice
```

### **Running Main Demo**
```bash
# Run with default matrix size
./10.cuda_basic/18.Thread_Hierarchy_Practice/18_Thread_Hierarchy_Practice

# Run with custom size
./10.cuda_basic/18.Thread_Hierarchy_Practice/18_Thread_Hierarchy_Practice 1024
```

### **Running Tests**
```bash
# Run all tests
ctest -R 18_Thread_Hierarchy

# Run with verbose output
./10.cuda_basic/18.Thread_Hierarchy_Practice/18_Thread_Hierarchy_Practice_test
```

---

## **8.7 Profiling and Analysis**

### **Thread Hierarchy Analysis**
```bash
# Analyze thread configurations
make 18_Thread_Hierarchy_Practice_thread_analysis
```

### **Warp Divergence Analysis**
```bash
# Check for warp divergence
make 18_Thread_Hierarchy_Practice_divergence_analysis
```

### **Occupancy Analysis**
```bash
# Analyze kernel occupancy
make 18_Thread_Hierarchy_Practice_occupancy_analysis
```

### **Tile Size Comparison**
```bash
# Compare different tile sizes
make 18_Thread_Hierarchy_Practice_tile_comparison
```

### **Thread Coarsening Analysis**
```bash
# Analyze coarsening benefits
make 18_Thread_Hierarchy_Practice_coarsening_analysis
```

---

## **8.8 Expected Output**

```
Using device: NVIDIA TITAN RTX
Compute capability: 7.5

=== Thread Hierarchy Analysis ===
Device Limits:
  Max threads per block: 1024
  Max thread dimensions: (1024, 1024, 64)
  Max grid dimensions: (2147483647, 65535, 65535)
  Warp size: 32
  Max warps per SM: 64
  Max blocks per SM: 32

=== Occupancy Analysis ===
Block size   64: Occupancy = 50.0% (Max active blocks: 32)
Block size  128: Occupancy = 100.0% (Max active blocks: 16)
Block size  256: Occupancy = 100.0% (Max active blocks: 8)
Block size  512: Occupancy = 100.0% (Max active blocks: 4)
Block size 1024: Occupancy = 100.0% (Max active blocks: 2)
Optimal block size suggested: 256

=== Warp Divergence Impact ===
With warp divergence: 1.234 ms
Without warp divergence: 0.456 ms
Speedup from avoiding divergence: 2.7x

=== Thread Hierarchy Benchmark ===
Matrix size: 512 x 512

Kernel Performance:
-----------------------------------------------------------------
                          Tiled 8x8:   12.345 ms,  43.21 GFLOPS
                        Tiled 16x16:    8.234 ms,  64.78 GFLOPS
                        Tiled 32x32:    9.123 ms,  58.45 GFLOPS
             Rectangular tiles 8x32:    7.892 ms,  67.54 GFLOPS
                      Warp optimized:    6.789 ms,  78.52 GFLOPS

Verifying correctness...
Results verified: PASS

=== Thread Hierarchy Demo Complete ===
```

---

## **8.9 Performance Guidelines**

### **Block Size Selection**

| Block Size | Pros | Cons | Best For |
|------------|------|------|----------|
| **64-128** | High occupancy potential | Low parallelism | Memory-bound kernels |
| **256** | Balanced | Standard choice | General purpose |
| **512-1024** | Maximum parallelism | May limit occupancy | Compute-bound kernels |

### **Tile Size Impact**

| Tile Size | Shared Memory | Occupancy | Performance |
|-----------|---------------|-----------|-------------|
| **8x8** | 256 bytes | High | Good for small matrices |
| **16x16** | 1 KB | Balanced | General purpose |
| **32x32** | 4 KB | May be limited | Large matrices |

---

## **8.10 Common Issues and Solutions**

### **Problem 1: Low Occupancy**
**Symptoms**: Poor GPU utilization
**Solutions**:
- Reduce register usage
- Decrease shared memory per block
- Adjust block size

### **Problem 2: Warp Divergence**
**Symptoms**: Poor performance in conditional code
**Solutions**:
- Reorganize conditions to align with warp boundaries
- Use warp-level primitives
- Restructure algorithm

### **Problem 3: Unbalanced Workload**
**Symptoms**: Some blocks finish much earlier
**Solutions**:
- Use dynamic scheduling
- Implement work stealing
- Better work distribution

### **Problem 4: Register Spilling**
**Symptoms**: Local memory usage in profiler
**Solutions**:
- Reduce variables per thread
- Use shared memory for temporary storage
- Enable `-maxrregcount` compiler flag

---

## **8.11 Optimization Techniques**

### **1. Persistent Threads**
```cpp
__global__ void persistent_kernel(int* work_queue) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int work_item;

    while ((work_item = atomicAdd(work_queue, 1)) < total_work) {
        // Process work_item
    }
}
```

### **2. Dynamic Parallelism**
```cpp
__global__ void parent_kernel() {
    if (complex_condition) {
        child_kernel<<<grid, block>>>();
    }
}
```

### **3. Grid-Stride Loops**
```cpp
__global__ void grid_stride_kernel(int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        // Process element i
    }
}
```

---

## **8.12 Exercises**

### **Exercise 1: Optimal Tile Size**
Find the optimal tile size for your GPU:
```cpp
// Test tile sizes: 4, 8, 12, 16, 20, 24, 28, 32
// Measure performance and occupancy
```

### **Exercise 2: Warp Specialization**
Implement a kernel where different warps perform different tasks:
```cpp
int warp_id = threadIdx.x / 32;
if (warp_id == 0) {
    // Data loading
} else if (warp_id == 1) {
    // Computation
} else {
    // Data storing
}
```

### **Exercise 3: Occupancy Tuning**
Use `__launch_bounds__` to control occupancy:
```cpp
__global__ void
__launch_bounds__(256, 8)  // max threads, min blocks
optimized_kernel() { }
```

### **Exercise 4: Thread Coarsening Study**
Implement varying coarsening factors and measure:
- Performance impact
- Register usage
- Occupancy changes

---

## **8.13 Best Practices**

1. **Start with 256 threads per block** - Good default for most kernels
2. **Align block dimensions with warp size** - Use multiples of 32
3. **Minimize warp divergence** - Keep conditional paths aligned
4. **Balance resources** - Don't max out one resource
5. **Profile before optimizing** - Identify actual bottlenecks
6. **Consider problem size** - Small problems may need different strategies
7. **Test multiple configurations** - Optimal settings vary by GPU

---

## **8.14 Advanced Topics**

### **Cooperative Groups**
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void coop_kernel() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp-level operations
    int sum = cg::reduce(warp, value, cg::plus<int>());
}
```

### **Tensor Cores (Volta+)**
```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm() {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Load, compute, store using tensor cores
}
```

---

## **8.15 Profiler Metrics**

### **Key Metrics to Monitor**
- `sm__threads_launched.sum`: Total threads launched
- `sm__warps_launched.sum`: Total warps launched
- `sm__maximum_warps_per_active_cycle_pct`: Warp occupancy
- `smsp__thread_inst_executed_per_inst_executed.ratio`: Thread efficiency
- `smsp__branch_targets_threads_divergent.pct`: Divergence percentage

---

## **8.16 References**

- [CUDA Programming Guide - Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- [Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
- [Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Cooperative Groups](https://developer.nvidia.com/blog/cooperative-groups/)

---

## ✅ **Summary**

This section demonstrated:
- Thread hierarchy organization and limits
- Various tiling strategies and their trade-offs
- Warp-level optimization techniques
- Occupancy analysis and tuning
- Avoiding warp divergence
- Thread coarsening benefits

**Key Takeaways:**
- Thread organization significantly impacts performance
- Optimal configuration depends on kernel characteristics
- Warp divergence can severely impact performance
- Occupancy is important but not the only factor
- Profile-guided optimization is essential

---

📄 **Next**: Part 9 - CUDA Memory API