# 🎯 Part 24: Memory Coalescing and Bank Conflicts

**Goal**: Optimize memory access patterns for maximum bandwidth utilization and minimal conflicts.

---

## **24.1 Overview**

Memory access patterns critically impact CUDA performance. This section covers:
- **Global memory coalescing** for efficient DRAM access
- **Shared memory bank conflicts** and mitigation strategies
- **Memory access pattern optimization**
- **Performance profiling and analysis**

### **Performance Impact**

| Access Pattern | Bandwidth Efficiency | Relative Performance |
|----------------|---------------------|---------------------|
| **Perfectly Coalesced** | 95-100% | 1.0x (baseline) |
| **Misaligned** | 80-90% | 0.8-0.9x |
| **Strided (2)** | 40-50% | 0.4-0.5x |
| **Random** | 3-10% | 0.03-0.1x |

---

## **24.2 Memory Coalescing Fundamentals**

### **24.2.1 What is Memory Coalescing?**

Memory coalescing combines memory accesses from multiple threads into fewer transactions:
- **Warp** = 32 threads executing in lockstep
- **Memory transaction** = 32, 64, or 128 byte segments
- **Goal**: One transaction per warp access

### **24.2.2 Coalescing Requirements**

For optimal coalescing, threads in a warp should:
1. Access **consecutive** memory addresses
2. Start at **aligned** addresses (32/64/128 byte boundaries)
3. Access same **data size** (1, 2, 4, 8, or 16 bytes)

### **24.2.3 Memory Access Patterns**

#### **Perfect Coalescing**
```cuda
// Each thread accesses consecutive 4-byte elements
__global__ void perfectCoalescing(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = data[tid];  // Threads 0-31 access data[0-31]
    }
}
```

#### **Strided Access (Poor Coalescing)**
```cuda
// Threads access non-consecutive elements
__global__ void stridedAccess(float* data, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * stride < n) {
        float value = data[tid * stride];  // Multiple transactions needed
    }
}
```

#### **Random Access (Worst Case)**
```cuda
// Each thread accesses random location
__global__ void randomAccess(float* data, int* indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = data[indices[tid]];  // Up to 32 transactions
    }
}
```

---

## **24.3 Optimizing Global Memory Access**

### **24.3.1 Structure of Arrays (SoA) vs Array of Structures (AoS)**

#### **Array of Structures (AoS) - Poor Coalescing**
```cuda
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void updateAoS(Particle* particles, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Poor coalescing - accessing non-consecutive x values
        particles[tid].x += particles[tid].vx;
        particles[tid].y += particles[tid].vy;
        particles[tid].z += particles[tid].vz;
    }
}
```

#### **Structure of Arrays (SoA) - Good Coalescing**
```cuda
struct ParticlesSoA {
    float* x;  float* y;  float* z;
    float* vx; float* vy; float* vz;
};

__global__ void updateSoA(ParticlesSoA particles, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Perfect coalescing - consecutive access
        particles.x[tid] += particles.vx[tid];
        particles.y[tid] += particles.vy[tid];
        particles.z[tid] += particles.vz[tid];
    }
}
```

### **24.3.2 2D Array Access Patterns**

#### **Row-Major Access (Coalesced)**
```cuda
__global__ void rowMajorAccess(float* matrix, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        // Coalesced - threads in same warp access consecutive columns
        float value = matrix[row * width + col];
    }
}
```

#### **Column-Major Access (Non-Coalesced)**
```cuda
__global__ void columnMajorAccess(float* matrix, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        // Non-coalesced - threads access same column, different rows
        float value = matrix[col * height + row];
    }
}
```

### **24.3.3 Alignment and Padding**

```cuda
// Ensure aligned access
__global__ void alignedAccess(float* data, int offset) {
    // Align offset to 128-byte boundary
    int aligned_offset = (offset + 31) & ~31;  // Round up to 32 floats

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[aligned_offset + tid];
}

// Padding for 2D arrays
__global__ void paddedMatrix(float* matrix, int width, int height, int pitch) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        // Use pitch for proper alignment
        float* row_ptr = (float*)((char*)matrix + row * pitch);
        float value = row_ptr[col];
    }
}
```

---

## **24.4 Shared Memory Bank Conflicts**

### **24.4.1 Bank Organization**

Shared memory is divided into 32 banks:
- Each bank is 4 bytes wide
- Successive 4-byte words map to successive banks
- Bank = address / 4 % 32

### **24.4.2 Types of Bank Conflicts**

#### **No Conflict - Linear Access**
```cuda
__shared__ float sdata[256];
// Thread 0 -> Bank 0, Thread 1 -> Bank 1, etc.
float value = sdata[threadIdx.x];
```

#### **2-Way Conflict**
```cuda
__shared__ float sdata[256];
// Threads 0 and 16 both access Bank 0
float value = sdata[threadIdx.x * 2];
```

#### **N-Way Conflict**
```cuda
__shared__ float sdata[256];
// Multiple threads access same bank
float value = sdata[threadIdx.x * stride];
// stride = 32 causes 32-way conflict (worst case)
```

### **24.4.3 Avoiding Bank Conflicts**

#### **Padding Arrays**
```cuda
// Without padding - conflicts in column access
__shared__ float tile[32][32];

// With padding - no conflicts
__shared__ float tile[32][33];  // or [32][32+1]

__global__ void matrixTranspose() {
    __shared__ float tile[32][33];

    // Load with coalescing
    tile[threadIdx.y][threadIdx.x] = input[...];
    __syncthreads();

    // Store transposed without bank conflicts
    output[...] = tile[threadIdx.x][threadIdx.y];
}
```

#### **Permutation Techniques**
```cuda
// XOR permutation to avoid conflicts
__global__ void permutedAccess() {
    __shared__ float sdata[1024];

    // Original access with conflicts
    // int idx = (threadIdx.x * STRIDE) % 1024;

    // Permuted access without conflicts
    int idx = threadIdx.x ^ (threadIdx.x / 32);
    float value = sdata[idx * STRIDE % 1024];
}
```

---

## **24.5 Advanced Optimization Techniques**

### **24.5.1 Texture Memory for Random Access**

```cuda
// Texture object for cached random access
texture<float, 1, cudaReadModeElementType> tex;

__global__ void textureRandomAccess(float* output, int* indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Cached through texture cache
        output[tid] = tex1Dfetch(tex, indices[tid]);
    }
}
```

### **24.5.2 Vectorized Memory Access**

```cuda
// Use vector types for wider loads
__global__ void vectorizedAccess(float4* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n/4) {
        // Load 16 bytes at once
        float4 values = data[tid];
        // Process values.x, values.y, values.z, values.w
    }
}

// Using built-in vector types
__global__ void vectorTypes(float2* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float2 val = data[tid];  // 8-byte load
    float sum = val.x + val.y;
}
```

### **24.5.3 Memory Access Optimization Pipeline**

```cuda
template<int BLOCK_SIZE>
__global__ void optimizedPipeline(float* output, float* input, int n) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Stage 1: Coalesced global read
    float value = (gid < n) ? input[gid] : 0.0f;

    // Stage 2: Shared memory for data reuse
    smem[tid] = value;
    __syncthreads();

    // Stage 3: Process in shared memory (any access pattern)
    if (tid > 0 && tid < BLOCK_SIZE - 1) {
        value = 0.25f * smem[tid - 1] + 0.5f * smem[tid] + 0.25f * smem[tid + 1];
    }
    __syncthreads();

    // Stage 4: Coalesced global write
    if (gid < n) {
        output[gid] = value;
    }
}
```

---

## **24.6 Performance Analysis and Profiling**

### **24.6.1 Metrics to Monitor**

```bash
# Global memory efficiency
ncu --metrics gld_efficiency,gst_efficiency ./program

# Memory throughput
ncu --metrics dram_read_throughput,dram_write_throughput ./program

# Transaction analysis
ncu --metrics gld_transactions,gst_transactions ./program

# Shared memory analysis
ncu --metrics shared_efficiency,shared_bank_conflicts ./program
```

### **24.6.2 Benchmark Implementation**

```cuda
__global__ void benchmarkCoalescing(float* data, int n, int pattern) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    clock_t start = clock();

    float sum = 0.0f;
    switch(pattern) {
        case 0:  // Coalesced
            for (int i = 0; i < 100; i++) {
                if (tid < n) sum += data[tid];
            }
            break;
        case 1:  // Strided
            for (int i = 0; i < 100; i++) {
                if (tid * 2 < n) sum += data[tid * 2];
            }
            break;
        case 2:  // Random
            for (int i = 0; i < 100; i++) {
                if (tid < n) sum += data[(tid * 12345) % n];
            }
            break;
    }

    clock_t end = clock();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Pattern %d: %d cycles\n", pattern, (int)(end - start));
    }
}
```

---

## **24.7 Example Programs**

### **24.7.1 Memory Coalescing Comparison**

File: `coalescing_demo.cu`
```cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define NUM_ELEMENTS (1024 * 1024)

// Coalesced access pattern
__global__ void coalescedAccess(float* data, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = data[tid] * 2.0f;
    }
}

// Strided access pattern
__global__ void stridedAccess(float* data, float* output, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    if (idx < n) {
        output[tid] = data[idx] * 2.0f;
    }
}

// Misaligned access pattern
__global__ void misalignedAccess(float* data, float* output, int n, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid + offset < n) {
        output[tid] = data[tid + offset] * 2.0f;
    }
}

void benchmark(const char* name, void (*kernel)(float*, float*, int, ...),
               float* d_input, float* d_output, int n, int extra_param = 0) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    if (extra_param > 0) {
        void (*kernel_with_param)(float*, float*, int, int) =
            (void (*)(float*, float*, int, int))kernel;
        kernel_with_param<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n, extra_param);
    } else {
        void (*kernel_simple)(float*, float*, int) =
            (void (*)(float*, float*, int))kernel;
        kernel_simple<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        if (extra_param > 0) {
            void (*kernel_with_param)(float*, float*, int, int) =
                (void (*)(float*, float*, int, int))kernel;
            kernel_with_param<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n, extra_param);
        } else {
            void (*kernel_simple)(float*, float*, int) =
                (void (*)(float*, float*, int))kernel;
            kernel_simple<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float bandwidth = (2.0f * n * sizeof(float) * 100) / (milliseconds * 1e6);
    printf("%s: %.2f ms, Bandwidth: %.2f GB/s\n", name, milliseconds, bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(NUM_ELEMENTS * sizeof(float));
    h_output = (float*)malloc(NUM_ELEMENTS * sizeof(float));

    // Initialize data
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc(&d_input, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&d_output, NUM_ELEMENTS * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== Memory Coalescing Benchmark ===\n");
    printf("Elements: %d\n\n", NUM_ELEMENTS);

    // Run benchmarks
    benchmark("Coalesced Access", (void (*)(float*, float*, int, ...))coalescedAccess,
              d_input, d_output, NUM_ELEMENTS);

    benchmark("Strided Access (2)", (void (*)(float*, float*, int, ...))stridedAccess,
              d_input, d_output, NUM_ELEMENTS / 2, 2);

    benchmark("Strided Access (8)", (void (*)(float*, float*, int, ...))stridedAccess,
              d_input, d_output, NUM_ELEMENTS / 8, 8);

    benchmark("Misaligned (1)", (void (*)(float*, float*, int, ...))misalignedAccess,
              d_input, d_output, NUM_ELEMENTS - 1, 1);

    benchmark("Misaligned (17)", (void (*)(float*, float*, int, ...))misalignedAccess,
              d_input, d_output, NUM_ELEMENTS - 17, 17);

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

### **24.7.2 Bank Conflict Demonstration**

File: `bank_conflicts.cu`
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define SHARED_SIZE 1024

// No bank conflicts
__global__ void noBankConflicts() {
    __shared__ float sdata[SHARED_SIZE];

    int tid = threadIdx.x;

    // Linear access - no conflicts
    sdata[tid] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += sdata[tid];
    }

    if (tid == 0) {
        sdata[0] = sum;  // Prevent optimization
    }
}

// 2-way bank conflicts
__global__ void twowayBankConflicts() {
    __shared__ float sdata[SHARED_SIZE];

    int tid = threadIdx.x;

    // Stride-2 access - 2-way conflicts
    sdata[tid * 2] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += sdata[tid * 2];
    }

    if (tid == 0) {
        sdata[0] = sum;
    }
}

// 8-way bank conflicts
__global__ void eightwayBankConflicts() {
    __shared__ float sdata[SHARED_SIZE];

    int tid = threadIdx.x;

    // Stride-8 access - 8-way conflicts
    int idx = (tid * 8) % SHARED_SIZE;
    sdata[idx] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += sdata[idx];
    }

    if (tid == 0) {
        sdata[0] = sum;
    }
}

// Padding to avoid conflicts
__global__ void paddingNoBankConflicts() {
    __shared__ float sdata[32][33];  // Padded array

    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    // Column access without conflicts due to padding
    sdata[col][row] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += sdata[col][row];
    }

    if (tid == 0) {
        sdata[0][0] = sum;
    }
}

void benchmarkKernel(const char* name, void (*kernel)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    kernel<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        kernel<<<1, BLOCK_SIZE>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s: %.3f ms\n", name, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Bank Conflict Benchmark ===\n\n");

    benchmarkKernel("No Bank Conflicts", noBankConflicts);
    benchmarkKernel("2-way Bank Conflicts", twowayBankConflicts);
    benchmarkKernel("8-way Bank Conflicts", eightwayBankConflicts);
    benchmarkKernel("Padding (No Conflicts)", paddingNoBankConflicts);

    printf("\nNote: Times are relative. Lower is better.\n");

    return 0;
}
```

---

## **24.8 Building and Running**

### **CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.18)
project(24_MemoryCoalescingBankConflicts CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Enable verbose PTX compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Coalescing demonstration
add_executable(coalescing_demo coalescing_demo.cu)

# Bank conflicts demonstration
add_executable(bank_conflicts bank_conflicts.cu)

# Structure comparison
add_executable(aos_vs_soa aos_vs_soa.cu)

# Memory pattern analysis
add_executable(memory_patterns memory_patterns.cu)

# Profiling targets
if(ENABLE_PROFILING)
    add_custom_target(profile_coalescing
        COMMAND ncu --metrics gld_efficiency,gst_efficiency ./coalescing_demo
        DEPENDS coalescing_demo
        COMMENT "Profiling memory coalescing efficiency"
    )

    add_custom_target(profile_banks
        COMMAND ncu --metrics shared_efficiency,bank_conflicts ./bank_conflicts
        DEPENDS bank_conflicts
        COMMENT "Profiling shared memory bank conflicts"
    )
endif()
```

### **Build and Run**
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
ninja

# Run demonstrations
./coalescing_demo
./bank_conflicts
./aos_vs_soa
./memory_patterns

# Profile memory access
ncu --metrics gld_efficiency,gst_efficiency,gld_transactions ./coalescing_demo
ncu --metrics shared_efficiency,shared_load_bank_conflicts ./bank_conflicts

# Detailed analysis
nsys profile --stats=true ./coalescing_demo
```

---

## **24.9 Optimization Guidelines**

### **24.9.1 Global Memory**

1. **Ensure coalesced access** - consecutive threads access consecutive addresses
2. **Use Structure of Arrays (SoA)** for better coalescing
3. **Align data structures** to cache line boundaries
4. **Use vector types** (float2, float4) for wider loads
5. **Consider texture memory** for random access patterns

### **24.9.2 Shared Memory**

1. **Pad arrays** to avoid bank conflicts
2. **Use permutation** for strided access patterns
3. **Minimize bank conflicts** in transpose operations
4. **Profile actual conflicts** with ncu
5. **Balance shared memory usage** with occupancy

### **24.9.3 General Best Practices**

1. **Profile first** - measure actual performance
2. **Optimize critical paths** - focus on bottlenecks
3. **Test different configurations** - block sizes, memory layouts
4. **Document access patterns** - for maintenance
5. **Consider memory hierarchy** - L1, L2, texture cache

---

## **24.10 Common Issues and Solutions**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Poor Coalescing** | Low bandwidth utilization | Restructure data layout (AoS → SoA) |
| **Bank Conflicts** | Slow shared memory access | Add padding or permute access |
| **Misalignment** | Reduced efficiency | Align to 128-byte boundaries |
| **Random Access** | Very low bandwidth | Use texture cache or shared memory |
| **Strided Access** | Multiple transactions | Transpose data or change algorithm |

---

## **24.11 Exercises**

1. **Optimize matrix transpose** - eliminate all bank conflicts
2. **Convert AoS to SoA** - measure performance improvement
3. **Implement histogram** with optimal memory patterns
4. **Profile and fix** coalescing issues in provided code
5. **Design access pattern** for 3D stencil computation

---

## **24.12 References**

- [CUDA C++ Best Practices - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations)
- [Global Memory Access Patterns](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)

---

**Next**: [Part 25: Dynamic Parallelism](../25.Dynamic_Parallelism/README.md)