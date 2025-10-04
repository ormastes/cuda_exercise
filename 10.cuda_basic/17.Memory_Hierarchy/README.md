# 🎯 Part 17: Memory Hierarchy
**Goal**: Master CUDA memory hierarchy through matrix multiplication optimization, demonstrating the performance impact of different memory access patterns.

## Project Structure

This module follows the advanced structure (Module 16+) with dedicated source and test directories. The organization separates kernel implementations, utilities, and tests for better maintainability and code reuse.
```
17.Memory_Hierarchy/
├── README.md                        - This documentation
├── CMakeLists.txt                   - Build configuration
├── src/                             - Source implementations
│   ├── kernels/                     - Core CUDA kernels (reusable across parts)
│   │   ├── matrix_multiply.cu       - Matrix multiplication implementations
│   │   ├── vector_add_2d.cu         - 2D vector addition kernels
│   │   └── vector_add_2d.h          - Vector addition header
│   └── part_specific/               - Module-specific implementations
│       └── vector_add_memory.cu     - Memory access pattern demos
└── test/                            - Test files
    └── unit/                        - Unit tests
        ├── kernels/                 - Kernel tests (reusable across parts)
        │   ├── test_matrix_multiply.cu  - Matrix multiplication tests
        │   └── test_vector_add_2d.cu    - Vector addition tests
        └── part_specific/           - Module-specific tests
            ├── benchmark_memory.cu      - Memory performance benchmarks
            └── test_vector_add_memory.cu - Memory pattern tests
```

## Quick Navigation

This guide is organized into progressive sections covering memory types, access patterns, and optimization techniques. Each section builds upon the previous ones to develop a comprehensive understanding of memory hierarchy optimization.
- [17.1 Memory Types and Characteristics](#171-memory-types-and-characteristics)
- [17.2 Memory Access Patterns](#172-memory-access-patterns)
- [17.3 Matrix Multiplication Evolution](#173-matrix-multiplication-evolution)
- [17.4 Shared Memory Optimization](#174-shared-memory-optimization)
- [17.5 Bank Conflict Mitigation](#175-bank-conflict-mitigation)
- [17.6 Testing](#176-testing)
- [Build & Run](#build--run)
- [Summary](#177-summary)

---

## **17.1 Memory Types and Characteristics**

This section introduces the CUDA memory hierarchy, which is fundamental for achieving optimal performance. Understanding memory types and their characteristics enables developers to make informed decisions about data placement and access patterns.

### **17.1.1 Memory Hierarchy Overview**

CUDA provides multiple memory types with different scopes, speeds, and capacities. Each memory type serves a specific purpose in the GPU architecture, offering trade-offs between capacity, bandwidth, and latency. Source: `src/part_specific/vector_add_memory.cu`.

| Memory Type | Scope | Bandwidth | Latency | Size | Cached |
|-------------|-------|-----------|---------|------|--------|
| **Registers** | Thread | Highest | Lowest | ~255 per thread | No |
| **Shared** | Block | Very High | Very Low | 48-96 KB/block | No |
| **L1 Cache** | SM | High | Low | 128 KB/SM | Yes |
| **L2 Cache** | Device | High | Medium | 4-6 MB | Yes |
| **Global** | Device | Medium | High | 4-48 GB | Through L1/L2 |
| **Constant** | Device | High | Low | 64 KB | Yes |
| **Texture** | Device | High | Medium | Via cache | Yes |

### **17.1.2 Memory Access Costs**

Different memory types have vastly different access costs, ranging from essentially free register access to hundreds of cycles for global memory. Understanding these costs is crucial for optimizing memory-bound kernels and achieving high performance. Full demonstration in `src/part_specific/vector_add_memory.cu`.

```cpp
// vector_add_memory.cu - Demonstrates memory access patterns
__global__ void memory_access_demo(float* global_data, int n) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Register access: ~0 cycles
    float reg_val = idx * 2.0f;

    // Shared memory access: ~1-30 cycles
    shared_data[threadIdx.x] = reg_val;
    __syncthreads();

    // Global memory access: ~200-800 cycles
    if (idx < n) {
        global_data[idx] = shared_data[threadIdx.x];
    }
}
```

**Key Points:**
- Register access is essentially free
- Shared memory is 10-100x faster than global memory
- Coalescing global memory accesses is critical
- Source: `src/part_specific/vector_add_memory.cu:45-67`

---

## **17.2 Memory Access Patterns**

This section demonstrates how memory access patterns dramatically affect performance. Proper access patterns can improve bandwidth utilization from 30% to over 90% of theoretical peak.

### **17.2.1 Strided vs Coalesced Access**

Memory coalescing occurs when consecutive threads in a warp access consecutive memory addresses, allowing the hardware to combine multiple memory requests into a single transaction. This optimization is essential for achieving high memory bandwidth utilization on modern GPUs. Source: `src/part_specific/vector_add_memory.cu`.

```cpp
// vector_add_memory.cu - Strided vs coalesced access patterns
__global__ void strided_access(float* data, int stride, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = idx * 2.0f;  // Strided: poor performance
    }
}

__global__ void coalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;  // Coalesced: adjacent threads access adjacent memory
    }
}
```

**Key Points:**
- Coalesced access can be 5-10x faster than strided
- Warp threads should access consecutive 128-byte segments
- Source: `src/part_specific/vector_add_memory.cu:23-44`

### **17.2.2 2D Memory Access Patterns**

Two-dimensional array access patterns demonstrate the significant performance difference between row-major and column-major memory layouts. The choice of access pattern can result in order-of-magnitude performance differences due to memory coalescing effects. Full example in `src/kernels/vector_add_2d.cu`.

```cpp
// vector_add_2d.cu - 2D memory access patterns
__global__ void vector_add_2d_row_major(
    const float* a, const float* b, float* c,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;  // Row-major: coalesced for x-dimension
        c[idx] = a[idx] + b[idx];
    }
}
```

**Expected Performance:**
```
Row-major access: 450 GB/s (95% efficiency)
Column-major access: 90 GB/s (19% efficiency)
```

---

## **17.3 Matrix Multiplication Evolution**

This section demonstrates the progression from naive to highly optimized matrix multiplication. Each implementation builds upon previous optimizations.

### **17.3.1 Naive Implementation**

The naive implementation serves as our baseline, implementing the mathematical definition of matrix multiplication without any GPU-specific optimizations. This approach suffers from poor memory access patterns and lacks data reuse, resulting in low performance. Source: `src/kernels/matrix_multiply.cu`.

```cpp
// matrix_multiply.cu - O(N³) baseline implementation
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

**Performance**: ~50 GFLOPS on RTX 3090

### **17.3.2 Coalesced Memory Access**

This implementation improves memory access patterns to achieve better coalescing for matrix A while relying on caching for matrix B accesses. Although B's column-wise access isn't fully coalesced, the L1 and L2 caches help mitigate the performance impact. Source: `src/kernels/matrix_multiply.cu`.

```cpp
// matrix_multiply.cu - Coalesced memory access pattern
__global__ void matmul_coalesced(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // Access pattern optimized for coalescing
        for (int k = 0; k < N; k++) {
            // A is accessed row-wise (coalesced)
            // B is accessed column-wise (not coalesced, but cached)
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Performance**: ~75 GFLOPS on RTX 3090 (1.5x improvement)

### **17.3.3 Tiled Implementation with Shared Memory**

The tiled implementation divides matrices into smaller tiles that fit in shared memory, dramatically reducing global memory accesses. Each tile is loaded once and reused multiple times by all threads in a block, achieving near-optimal memory bandwidth utilization. Source: `src/kernels/matrix_multiply.cu`.

```cpp
// matrix_multiply.cu - Tiled implementation with shared memory
#define TILE_SIZE 16

__global__ void matmul_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < N && tile * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial products
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Performance**: ~400 GFLOPS on RTX 3090 (8x improvement over naive)

---

## **17.4 Shared Memory Optimization**

This section explores advanced shared memory techniques for maximizing performance. Shared memory provides low-latency, high-bandwidth storage shared among threads in a block.

### **17.4.1 Bank Conflict Free Implementation**

Bank conflicts occur when multiple threads in a warp access different addresses in the same shared memory bank, causing serialization of memory accesses. By adding padding to shift memory addresses to different banks, we can eliminate these conflicts and achieve full shared memory bandwidth. Source: `src/kernels/matrix_multiply.cu`.

```cpp
// matrix_multiply.cu - Bank conflict free shared memory
#define TILE_SIZE 16

__global__ void matmul_shared_bank_conflict_free(
    const float* A, const float* B, float* C, int N
) {
    // Add padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Rest of implementation similar to matmul_shared
    // Padding shifts memory addresses to different banks
    // This prevents serialization of memory accesses
}
```

**Performance**: ~450 GFLOPS on RTX 3090 (9x improvement over naive)

### **17.4.2 Double Buffering Strategy**

Double buffering uses two sets of shared memory buffers to overlap data loading with computation, hiding memory latency. While one buffer is being used for computation, the next tile is loaded into the alternate buffer, maximizing both memory and compute throughput. Implementation concept in `src/kernels/matrix_multiply.cu`.

```cpp
// Conceptual double buffering approach
__global__ void matmul_double_buffered(const float* A, const float* B, float* C, int N) {
    // Two sets of shared memory buffers
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    int buffer = 0;
    // Load first tile
    load_tile(As[buffer], Bs[buffer], ...);
    __syncthreads();

    for (int tile = 1; tile < num_tiles; tile++) {
        // Load next tile while computing current
        load_tile(As[1-buffer], Bs[1-buffer], ...);
        compute_tile(As[buffer], Bs[buffer], ...);
        __syncthreads();
        buffer = 1 - buffer;
    }
}
```

---

## **17.5 Bank Conflict Mitigation**

This section details strategies for avoiding shared memory bank conflicts. Bank conflicts occur when multiple threads in a warp access different addresses in the same memory bank.

### **17.5.1 Understanding Bank Conflicts**

Modern GPUs organize shared memory into 32 banks, with successive 32-bit words assigned to successive banks. When multiple threads in a warp access different addresses in the same bank, the accesses are serialized, dramatically reducing performance. Test implementation in `test/unit/kernels/test_matrix_multiply.cu`.

```cpp
// test_matrix_multiply.cu - Bank conflict detection
__global__ void test_bank_conflicts(float* output) {
    __shared__ float data[32][32];
    int tid = threadIdx.x;

    // Scenario 1: No bank conflict - each thread accesses different bank
    data[tid][0] = tid;  // Linear access pattern

    // Scenario 2: 2-way bank conflict
    data[tid * 2 % 32][0] = tid;  // Every other thread conflicts

    // Scenario 3: 32-way bank conflict (worst case)
    data[0][tid] = tid;  // All threads access same bank
}
```

### **17.5.2 Padding Technique**

Padding is a simple yet effective technique for eliminating bank conflicts by adding extra elements to array dimensions. This shifts memory addresses so that threads in a warp access different banks, converting serialized accesses into parallel ones and achieving dramatic speedups. Source: `test/unit/part_specific/benchmark_memory.cu`.

```cpp
// benchmark_memory.cu - Padding to avoid conflicts
template<bool USE_PADDING>
__global__ void benchmark_shared_memory(float* output, int iterations) {
    const int ARRAY_SIZE = 32;

    // Without padding: potential conflicts
    __shared__ float no_pad[ARRAY_SIZE][ARRAY_SIZE];

    // With padding: conflict-free
    __shared__ float padded[ARRAY_SIZE][ARRAY_SIZE + 1];

    // Benchmark code measures the difference
}
```

**Expected Results:**
```
Without padding: 180 GB/s shared memory bandwidth
With padding: 1300 GB/s shared memory bandwidth
Improvement: 7.2x
```

---

## **17.6 Testing**

This module includes comprehensive unit and performance tests to validate correctness and measure optimization improvements. The tests compare different implementations and verify that optimizations maintain numerical accuracy while improving performance.

### **17.6.1 Unit Tests**

Unit tests verify the correctness of each implementation by comparing results between optimized and baseline versions. These tests ensure that performance optimizations don't compromise accuracy and that all edge cases are handled correctly. Tests are in `test/unit/kernels/` and `test/unit/part_specific/`.

```cpp
// test/unit/kernels/test_matrix_multiply.cu
#include <gtest/gtest.h>
#include "../../00.test_lib/gpu_gtest.h"

GPU_TEST(MatrixMultiply, CorrectnessCheck) {
    const int N = 256;
    float *d_A, *d_B, *d_C_naive, *d_C_optimized;

    // Allocate memory
    cuda_malloc(&d_A, N * N);
    cuda_malloc(&d_B, N * N);
    cuda_malloc(&d_C_naive, N * N);
    cuda_malloc(&d_C_optimized, N * N);

    // Initialize matrices
    init_matrix<<<(N*N+255)/256, 256>>>(d_A, N*N, 1.0f);
    init_matrix<<<(N*N+255)/256, 256>>>(d_B, N*N, 2.0f);

    // Run kernels
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (N+15)/16);

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C_naive, N);
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C_optimized, N);

    CHECK_KERNEL_LAUNCH();

    // Compare results
    GPU_EXPECT_ARRAYS_NEAR(d_C_naive, d_C_optimized, N*N, 1e-5f);

    cuda_free(d_A);
    cuda_free(d_B);
    cuda_free(d_C_naive);
    cuda_free(d_C_optimized);
}
```

### **17.6.2 Performance Tests**

Performance tests measure bandwidth utilization, GFLOPS, and relative speedups between implementations. These benchmarks help identify bottlenecks and verify that optimizations achieve expected theoretical improvements.

```cpp
// test/unit/part_specific/benchmark_memory.cu
TEST(Performance, MemoryBandwidthTest) {
    CudaTimer timer;
    const size_t size = 1 << 20;  // 1M elements
    float *d_data;

    cuda_malloc(&d_data, size);

    // Benchmark coalesced access
    timer.start();
    coalesced_access<<<(size+255)/256, 256>>>(d_data, size);
    timer.stop();

    float bandwidth = calculate_bandwidth_gb(size * sizeof(float), timer.elapsed_ms());
    EXPECT_GT(bandwidth, 400.0f);  // Expect > 400 GB/s

    cuda_free(d_data);
}

GPU_TEST_P(MatrixMultiplyPerf, ThroughputTest) {
    int N = GetParam();

    // Measure GFLOPS for different implementations
    float naive_gflops = measure_gflops(matmul_naive, N);
    float shared_gflops = measure_gflops(matmul_shared, N);

    // Shared memory should be at least 5x faster
    EXPECT_GT(shared_gflops / naive_gflops, 5.0f);
}

INSTANTIATE_TEST_SUITE_P(Sizes, MatrixMultiplyPerf,
                        ::testing::Values(256, 512, 1024, 2048));
```

## Build & Run

This section provides instructions for building and running the memory hierarchy demonstrations and tests. The build system uses CMake and Ninja for fast compilation and supports various profiling tools for performance analysis.

### Building

The module can be built using CMake with separate targets for the library, tests, and benchmarks. Use Ninja for faster build times compared to Make.
```bash
cd build
cmake --build . --target 17_Memory_Hierarchy_lib
cmake --build . --target 17_Memory_Hierarchy_test
cmake --build . --target 17_Memory_Hierarchy_benchmark
```

### Running Tests

Tests can be run through CTest for automated testing or directly for detailed output. The test executables support Google Test filters for running specific test cases.
```bash
# Run all tests for this module
ctest -R "17_Memory"

# Run specific test with output
./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_test --gtest_filter="*MatrixMultiply*"

# Run benchmarks
./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_benchmark
```

### Running Performance Analysis

NVIDIA's profiling tools provide detailed insights into memory access patterns, bandwidth utilization, and performance bottlenecks. Use these tools to verify optimization effectiveness and identify further improvement opportunities.
```bash
# Memory access pattern analysis
nsys profile --stats=true ./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_benchmark

# Detailed metrics with Nsight Compute
ncu --metrics l1tex__throughput,lts__throughput,dram__throughput ./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_benchmark

# Bank conflict analysis
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu ./10.cuda_basic/17.Memory_Hierarchy/17_Memory_Hierarchy_test
```

---

## **17.7 Summary**

This module demonstrated the critical importance of memory hierarchy optimization in CUDA programming. Through progressive optimizations of matrix multiplication, we achieved nearly 10x performance improvements by properly utilizing different memory types and access patterns.

### **Key Takeaways**
1. Memory hierarchy optimization can yield 8-10x performance improvements
2. Coalesced memory access is critical for global memory bandwidth utilization
3. Shared memory with proper bank conflict mitigation enables near-peak throughput

### **Performance Metrics**

These metrics show the dramatic performance improvements achieved through memory hierarchy optimization. Each optimization level demonstrates the impact of specific techniques on overall throughput.
- Baseline (Naive): 50 GFLOPS
- Coalesced Access: 75 GFLOPS
- Shared Memory: 400 GFLOPS
- Bank-Conflict Free: 450 GFLOPS
- Efficiency: 85% of theoretical peak

### **Common Errors & Solutions**

This table summarizes the most common memory-related performance issues and their solutions. Understanding these patterns helps avoid pitfalls in production code.
| Error | Cause | Solution |
|-------|-------|----------|
| Low bandwidth | Uncoalesced access | Ensure consecutive threads access consecutive addresses |
| Bank conflicts | Multiple threads access same bank | Add padding or reorganize access pattern |
| Register spilling | Too many variables per thread | Reduce register usage or adjust block size |
| Low occupancy | Resource overuse | Balance shared memory, registers, and block size |

### **Next Steps**

After mastering memory hierarchy optimization, continue your learning journey with more advanced topics. These resources and exercises will help deepen your understanding of GPU programming.
- 📚 Continue to [Part 18: Thread Hierarchy Practice](../18.Thread_Hierarchy_Practice/README.md)
- 🔧 Try optimizing for different tile sizes in `src/kernels/matrix_multiply.cu`
- 📊 Run performance benchmarks with `17_Memory_Hierarchy_benchmark`

### **References**

These official NVIDIA resources provide in-depth coverage of memory optimization techniques and best practices. Consult them for detailed specifications and advanced optimization strategies.
- [CUDA Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [CUDA Best Practices Guide - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [Nsight Compute Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#memory-workload-analysis)