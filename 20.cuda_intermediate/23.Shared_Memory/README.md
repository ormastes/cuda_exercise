# 💾 Part 23: Shared Memory

**Goal**: Master shared memory usage for high-performance CUDA kernels through efficient data sharing and memory access patterns.

---

## **23.1 Overview**

Shared memory is a programmable cache that enables fast data sharing among threads within a block. It provides:
- **100x faster** access than global memory
- **User-controlled** caching strategy
- **Inter-thread communication** within blocks
- **Reduced global memory bandwidth** requirements

### **Memory Hierarchy**

| Memory Type | Scope | Latency | Bandwidth | Size |
|------------|-------|---------|-----------|------|
| **Registers** | Thread | ~0 cycles | Highest | 255 per thread |
| **Shared Memory** | Block | ~1-30 cycles | ~10 TB/s | 48-96 KB per SM |
| **L1 Cache** | SM | ~30 cycles | ~4 TB/s | 16-48 KB |
| **L2 Cache** | Device | ~200 cycles | ~2 TB/s | 4-6 MB |
| **Global Memory** | Device | ~400 cycles | ~900 GB/s | 8-80 GB |

---

## **23.2 Shared Memory Fundamentals**

### **23.2.1 Declaration and Allocation**

#### **Static Allocation**
```cuda
__global__ void kernel() {
    // Fixed size known at compile time
    __shared__ float sdata[256];

    // Multi-dimensional arrays
    __shared__ float tile[16][16];
}
```

#### **Dynamic Allocation**
```cuda
__global__ void kernel() {
    // Size specified at kernel launch
    extern __shared__ float sdata[];

    // Access shared memory
    sdata[threadIdx.x] = data[globalIdx];
}

// Launch with dynamic shared memory
kernel<<<blocks, threads, sharedMemSize>>>(...);
```

#### **Multiple Dynamic Arrays**
```cuda
__global__ void kernel(int n) {
    extern __shared__ float shared[];

    // Partition shared memory manually
    float* array1 = shared;
    float* array2 = &shared[n];
    int* array3 = (int*)&shared[2*n];
}
```

### **23.2.2 Memory Configuration**

```cuda
// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);

// Configure L1/Shared memory split (deprecated in newer GPUs)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
```

---

## **23.3 Classic Shared Memory Patterns**

### **23.3.1 Matrix Multiplication with Tiling**

```cuda
#define TILE_SIZE 16

__global__ void matrixMulShared(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Collaborative loading
        if (row < N && m * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + m * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute on tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### **23.3.2 1D Stencil Computation**

```cuda
__global__ void stencil1D(float* out, float* in, int N) {
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;

    // Load data with halo
    temp[lindex] = in[gindex];

    // Load halo elements
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = (gindex >= RADIUS) ?
            in[gindex - RADIUS] : 0.0f;
        temp[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < N) ?
            in[gindex + BLOCK_SIZE] : 0.0f;
    }

    __syncthreads();

    // Apply stencil
    float result = 0.0f;
    #pragma unroll
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[lindex + offset] * coeff[offset + RADIUS];
    }

    out[gindex] = result;
}
```

### **23.3.3 Parallel Reduction**

```cuda
__global__ void reduce(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load and perform first reduction
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n)
        mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp reduction
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] = mySum = mySum + smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] = mySum = mySum + smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] = mySum = mySum + smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] = mySum = mySum + smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] = mySum = mySum + smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] = mySum = mySum + smem[tid + 1];
    }

    // Write result
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

---

## **23.4 Bank Conflicts**

### **23.4.1 Understanding Bank Conflicts**

Shared memory is organized into **32 banks** (4-byte wide each). Bank conflicts occur when multiple threads in a warp access different addresses in the same bank.

#### **Conflict-Free Access Patterns**

```cuda
// Linear access - No conflicts
__shared__ float data[256];
float value = data[threadIdx.x];  // Each thread accesses different bank

// Stride-1 access - No conflicts
__shared__ float tile[32][33];  // Padding prevents conflicts
float value = tile[threadIdx.y][threadIdx.x];

// Broadcast - No conflicts
__shared__ float data[256];
float value = data[0];  // All threads read same address
```

#### **Patterns Causing Bank Conflicts**

```cuda
// 2-way bank conflict
__shared__ float data[256];
float value = data[threadIdx.x * 2];  // Even threads conflict

// 32-way bank conflict (worst case)
__shared__ float data[256];
float value = data[threadIdx.x * 32];  // All threads hit same bank

// Matrix transpose - causes conflicts without padding
__shared__ float tile[32][32];
float value = tile[threadIdx.x][threadIdx.y];  // Column access conflicts
```

### **23.4.2 Avoiding Bank Conflicts**

#### **Padding Technique**
```cuda
// Add padding to avoid conflicts in 2D arrays
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

__global__ void transposeNoBankConflicts(float* odata, float* idata, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load tile (coalesced)
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();

    // Write transposed (no bank conflicts due to padding)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    odata[y * width + x] = tile[threadIdx.x][threadIdx.y];
}
```

#### **Address Permutation**
```cuda
// Permute thread access pattern to avoid conflicts
__global__ void accessPattern() {
    __shared__ float sdata[256];

    // Original pattern with conflicts
    // int offset = threadIdx.x * STRIDE;

    // Permuted pattern without conflicts
    int offset = ((threadIdx.x + threadIdx.x / 32) * STRIDE) % 256;
    float value = sdata[offset];
}
```

---

## **23.5 Advanced Techniques**

### **23.5.1 Double Buffering**

```cuda
__global__ void pipelineKernel(float* output, float* input, int N) {
    __shared__ float buffer[2][BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initial load
    int current = 0;
    buffer[current][tid] = input[gid];
    __syncthreads();

    for (int i = 1; i < N / blockDim.x; i++) {
        int next = 1 - current;

        // Load next tile while processing current
        if (gid + i * blockDim.x < N) {
            buffer[next][tid] = input[gid + i * blockDim.x];
        }

        // Process current buffer
        float result = processData(buffer[current][tid]);
        output[gid + (i-1) * blockDim.x] = result;

        __syncthreads();
        current = next;
    }

    // Process last buffer
    output[gid + (N/blockDim.x - 1) * blockDim.x] =
        processData(buffer[current][tid]);
}
```

### **23.5.2 Shared Memory Atomics**

```cuda
__global__ void histogram(int* hist, int* data, int n) {
    __shared__ int smem_hist[256];

    // Initialize shared memory histogram
    if (threadIdx.x < 256) {
        smem_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Process data
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(&smem_hist[data[tid]], 1);
    }
    __syncthreads();

    // Merge to global histogram
    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], smem_hist[threadIdx.x]);
    }
}
```

---

## **23.6 Performance Optimization**

### **23.6.1 Optimization Guidelines**

1. **Maximize Occupancy**
   - Balance shared memory usage with thread count
   - Use dynamic allocation when size varies

2. **Minimize Bank Conflicts**
   - Use padding for 2D arrays
   - Permute access patterns for strided access

3. **Coalesce Global Access**
   - Load to shared memory with coalesced pattern
   - Process in shared memory with any pattern

4. **Hide Latency**
   - Use double buffering
   - Overlap computation with memory access

### **23.6.2 Performance Analysis**

```cuda
// Measure shared memory throughput
__global__ void benchmarkSharedMemory() {
    __shared__ float sdata[1024];

    clock_t start = clock();

    // Perform many shared memory operations
    #pragma unroll
    for (int i = 0; i < 1000; i++) {
        sdata[threadIdx.x] = sdata[threadIdx.x] * 2.0f + 1.0f;
    }

    clock_t end = clock();

    if (threadIdx.x == 0) {
        printf("Cycles: %d\n", (int)(end - start));
    }
}
```

---

## **23.7 Example Programs**

### **23.7.1 Matrix Transpose**

File: `matrix_transpose.cu`
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Naive transpose - bank conflicts
__global__ void transposeNaive(float* odata, float* idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Optimized transpose - no bank conflicts
__global__ void transposeOptimized(float* odata, float* idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // Padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### **23.7.2 1D Convolution**

File: `convolution_1d.cu`
```cuda
#define KERNEL_RADIUS 3
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

__constant__ float d_kernel[KERNEL_SIZE];

__global__ void convolution1D(float* output, float* input, int n) {
    __shared__ float s_data[BLOCK_SIZE + 2 * KERNEL_RADIUS];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x + KERNEL_RADIUS;

    // Load main data
    s_data[lid] = (gid < n) ? input[gid] : 0.0f;

    // Load halo
    if (threadIdx.x < KERNEL_RADIUS) {
        s_data[lid - KERNEL_RADIUS] =
            (gid >= KERNEL_RADIUS) ? input[gid - KERNEL_RADIUS] : 0.0f;
        s_data[lid + BLOCK_SIZE] =
            (gid + BLOCK_SIZE < n) ? input[gid + BLOCK_SIZE] : 0.0f;
    }

    __syncthreads();

    // Compute convolution
    float result = 0.0f;
    #pragma unroll
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        result += s_data[lid + i] * d_kernel[KERNEL_RADIUS + i];
    }

    if (gid < n) {
        output[gid] = result;
    }
}
```

---

## **23.8 Building and Running**

### **CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.18)
project(23_SharedMemory CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Matrix transpose example
add_executable(matrix_transpose matrix_transpose.cu)
target_compile_options(matrix_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>)

# Convolution example
add_executable(convolution_1d convolution_1d.cu)

# Reduction example
add_executable(reduction reduction.cu)

# Add profiling targets
if(ENABLE_PROFILING)
    add_custom_target(${PROJECT_NAME}_profile
        COMMAND nsys profile --stats=true ./matrix_transpose
        COMMAND ncu --metrics shared_efficiency ./matrix_transpose
        DEPENDS matrix_transpose
        COMMENT "Profiling shared memory usage"
    )
endif()
```

### **Build and Run**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ninja

# Run examples
./matrix_transpose
./convolution_1d
./reduction

# Profile shared memory
nsys profile --stats=true ./matrix_transpose
ncu --metrics shared_efficiency,shared_load_throughput,shared_store_throughput ./matrix_transpose
```

---

## **23.9 Common Pitfalls and Solutions**

### **Problem 1: Bank Conflicts**
**Symptom**: Poor shared memory throughput
**Solution**: Add padding or permute access patterns

### **Problem 2: Race Conditions**
**Symptom**: Incorrect results
**Solution**: Use proper synchronization with `__syncthreads()`

### **Problem 3: Insufficient Shared Memory**
**Symptom**: Kernel launch failure
**Solution**: Reduce per-block usage or use dynamic allocation

### **Problem 4: Low Occupancy**
**Symptom**: Poor GPU utilization
**Solution**: Balance shared memory usage with thread count

---

## **23.10 Best Practices**

1. **Always profile** shared memory efficiency
2. **Use padding** for 2D arrays to avoid bank conflicts
3. **Minimize synchronization** points
4. **Consider L1 cache** as an alternative for read-only data
5. **Use `__syncthreads()` correctly** - all threads must reach it
6. **Benchmark different configurations** for optimal performance
7. **Document shared memory usage** in kernels

---

## **23.11 Exercises**

1. **Implement 2D convolution** using shared memory tiling
2. **Optimize histogram computation** with shared memory atomics
3. **Create a prefix sum (scan)** algorithm using shared memory
4. **Implement bitonic sort** in shared memory
5. **Profile and eliminate bank conflicts** in provided kernels

---

## **23.12 References**

- [CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)
- [CUDA C++ Best Practices - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory)
- [Using Shared Memory in CUDA](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Optimizing Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

---

**Next**: [Part 24: Memory Coalescing and Bank Conflicts](../24.Memory_Coalescing_and_Bank_Conflicts/README.md)