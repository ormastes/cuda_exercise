/**
 * Thread Operations - Thread Hierarchy Practice (Part 18)
 *
 * Demonstrates thread-specific operations and optimizations:
 * - Warp divergence patterns
 * - Occupancy calculation and optimization
 * - Thread coalescing demonstrations
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cuda_utils.h"

namespace cg = cooperative_groups;

// =============================================================================
// Warp Divergence Demonstrations
// =============================================================================

/**
 * Bad divergence pattern - threads in same warp take different paths
 * Performance impact: ~50% throughput loss
 */
__global__ void divergent_kernel_bad(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Bad: Adjacent threads diverge
    if (threadIdx.x % 2 == 0) {
        // Even threads - complex computation
        data[idx] = idx * idx + __syncthreads_count(1);
        for (int i = 0; i < 10; i++) {
            data[idx] += i;
        }
    } else {
        // Odd threads - simple computation
        data[idx] = idx + 1;
    }
}

/**
 * Good divergence pattern - warps take uniform paths
 * Performance: Minimal divergence impact
 */
__global__ void divergent_kernel_good(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Good: Entire warps take same branch
    int warpId = threadIdx.x / 32;
    if (warpId % 2 == 0) {
        // Even warps - complex computation
        data[idx] = idx * idx;
        for (int i = 0; i < 10; i++) {
            data[idx] += i;
        }
    } else {
        // Odd warps - simple computation
        data[idx] = idx + 1;
    }
}

// =============================================================================
// Memory Coalescing Patterns
// =============================================================================

/**
 * Strided memory access - poor coalescing
 * Memory efficiency: ~25% on typical GPUs
 */
__global__ void strided_access(float* data, int stride, int n) {
    int tid = threadIdx.x;
    int idx = tid * stride;

    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

/**
 * Coalesced memory access - optimal pattern
 * Memory efficiency: ~95% on typical GPUs
 */
__global__ void coalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

// =============================================================================
// Occupancy Test Kernels
// =============================================================================

/**
 * High register pressure kernel
 * Used to test impact of register usage on occupancy
 */
template<int REGISTERS_PER_THREAD>
__global__ __launch_bounds__(1024, 2)
void high_register_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Use registers to control register pressure
    float regs[REGISTERS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < REGISTERS_PER_THREAD; i++) {
        regs[i] = data[idx] * (i + 1);
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < REGISTERS_PER_THREAD; i++) {
        sum += regs[i];
    }

    data[idx] = sum;
}

/**
 * Shared memory intensive kernel
 * Used to test impact of shared memory on occupancy
 */
__global__ void shared_memory_kernel(float* data, int n, int sharedBytes) {
    extern __shared__ float shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n && tid < sharedBytes / sizeof(float)) {
        shared[tid] = data[idx];
    }

    __syncthreads();

    if (idx < n && tid < sharedBytes / sizeof(float)) {
        float sum = 0.0f;
        for (int i = 0; i < min(32, (int)(sharedBytes / sizeof(float))); i++) {
            sum += shared[i];
        }
        data[idx] = sum;
    }
}

// =============================================================================
// Warp-level Primitives
// =============================================================================

/**
 * Warp shuffle operations for efficient communication
 * Avoids shared memory for intra-warp communication
 */
__global__ void warp_shuffle_sum(float* data, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    float value = (idx < n) ? data[idx] : 0.0f;

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    // Lane 0 has the sum for the warp
    if (laneId == 0 && idx < n) {
        output[blockIdx.x * (blockDim.x / 32) + warpId] = value;
    }
}

/**
 * Warp vote functions for consensus
 */
__global__ void warp_vote_kernel(int* data, int* results, int n, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = idx / 32;

    bool predicate = (idx < n) ? (data[idx] > threshold) : false;

    // All threads in warp meet condition
    if (__all_sync(0xFFFFFFFF, predicate)) {
        if (threadIdx.x % 32 == 0) {
            results[warpId] = 1;  // All passed
        }
    }
    // Any thread in warp meets condition
    else if (__any_sync(0xFFFFFFFF, predicate)) {
        if (threadIdx.x % 32 == 0) {
            results[warpId] = 0;  // Some passed
        }
    }
    else {
        if (threadIdx.x % 32 == 0) {
            results[warpId] = -1; // None passed
        }
    }
}

// Explicit template instantiations
template __global__ void high_register_kernel<8>(float*, int);
template __global__ void high_register_kernel<16>(float*, int);
template __global__ void high_register_kernel<32>(float*, int);
template __global__ void high_register_kernel<64>(float*, int);