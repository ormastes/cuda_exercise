// vector_add_2d.cu - Part 17: Memory Hierarchy Evolution
// Demonstrates memory hierarchy optimization techniques for 2D vector operations
#include "vector_add_2d.h"
#include <cstdio>
#include <cassert>
#include <cmath>

// ============================================================================
// Basic Implementation (from Part 12-16)
// ============================================================================

// Simple 2D vector addition kernel with global memory
__global__ void vector_add_2d(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;  // Row-major access (coalesced for consecutive x)

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// ============================================================================
// Part 17: Memory Hierarchy Optimizations
// ============================================================================

// Version 1: Optimized with shared memory tiling
__global__ void vector_add_2d_shared(const float* A, const float* B, float* C,
                                      int width, int height) {
    // Shared memory tile for current block
    extern __shared__ float tile[];
    float* tileA = tile;
    float* tileB = &tile[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int tid = ty * blockDim.x + tx;
    int gid = y * width + x;

    // Coalesced load from global to shared memory
    if (x < width && y < height) {
        tileA[tid] = A[gid];
        tileB[tid] = B[gid];
    }
    __syncthreads();

    // Compute from shared memory (demonstrates reuse potential)
    if (x < width && y < height) {
        float a_val = tileA[tid];
        C[gid] = square(a_val) + tileB[tid];
    }
}

// Version 2: Coalesced memory access pattern demonstration
__global__ void vector_add_2d_coalesced(const float* A, const float* B, float* C,
                                         int width, int height) {
    // Compute coordinates ensuring coalesced access
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Row-major indexing ensures threads in same warp access consecutive memory
        int idx = y * width + x;

        // All memory accesses are coalesced (consecutive threads -> consecutive addresses)
        float a = A[idx];  // Coalesced read
        float b = B[idx];  // Coalesced read
        C[idx] = square(a) + b;  // Coalesced write
    }
}

// Version 3: Using constant memory for frequently accessed scalar
__constant__ float c_scalar[1];  // Constant memory for broadcast

__global__ void vector_add_2d_constant(const float* A, const float* B, float* C,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // Constant memory broadcast is efficient when all threads read same value
        C[idx] = square(A[idx]) + B[idx] + c_scalar[0];
    }
}

// Version 4: Register-optimized with loop unrolling
__global__ void vector_add_2d_unrolled(const float* A, const float* B, float* C,
                                        int width, int height) {
    // Process multiple elements per thread using registers
    int x_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 4 elements per thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height) {
        int idx_base = y * width + x_base;

        // Unroll loop to keep values in registers
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int x = x_base + i;
            if (x < width) {
                int idx = idx_base + i;
                float a = A[idx];
                float b = B[idx];
                C[idx] = square(a) + b;
            }
        }
    }
}

// Version 5: Demonstrating bank conflict avoidance
__global__ void vector_add_2d_bank_conflict_free(const float* A, const float* B, float* C,
                                                   int width, int height) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float sA[16][16 + 1];  // +1 padding eliminates bank conflicts
    __shared__ float sB[16][16 + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int gid = y * width + x;

    // Load to shared memory
    if (x < width && y < height) {
        sA[ty][tx] = A[gid];
        sB[ty][tx] = B[gid];
    }
    __syncthreads();

    // Compute (padding ensures no bank conflicts)
    if (x < width && y < height) {
        C[gid] = square(sA[ty][tx]) + sB[ty][tx];
    }
}

// ============================================================================
// Reduction Operations (from Part 16 with memory optimizations)
// ============================================================================

__global__ void reduce_sum(const float* input, float* output, int N, int stride) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Coalesced load with grid-stride loop
    float sum = 0.0f;
    while (i < N) {
        sum += input[i];
        if (i + blockDim.x < N)
            sum += input[i + blockDim.x];
        i += gridDim.x * blockDim.x * 2;
    }

    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory (memory-optimized)
    if (blockDim.x >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* vdata = sdata;
        if (blockDim.x >= 64) vdata[tid] += vdata[tid + 32];
        if (blockDim.x >= 32) vdata[tid] += vdata[tid + 16];
        if (blockDim.x >= 16) vdata[tid] += vdata[tid + 8];
        if (blockDim.x >= 8) vdata[tid] += vdata[tid + 4];
        if (blockDim.x >= 4) vdata[tid] += vdata[tid + 2];
        if (blockDim.x >= 2) vdata[tid] += vdata[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================================
// Memory Access Pattern Demonstrations
// ============================================================================

// Strided access (poor coalescing) - for comparison
__global__ void vector_add_2d_strided(const float* A, const float* B, float* C,
                                       int width, int height, int stride) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // Strided access - poor memory coalescing
        C[idx] = square(A[idx]) + B[idx];
    }
}

// ============================================================================
// Debugging Kernels (from Part 16)
// ============================================================================

__global__ void vector_add_2d_with_bug(const float* a, const float* b, float* c,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // BUG: Missing boundary check
    int idx = y * width + x;
    c[idx] = a[idx] + b[idx];
}

__global__ void vector_add_2d_with_assertion(const float* a, const float* b, float* c,
                                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    assert(x < width && y < height && "Index out of bounds!");

    if (x < width && y < height) {
        c[idx] = square(a[idx]) + b[idx];
    }
}
