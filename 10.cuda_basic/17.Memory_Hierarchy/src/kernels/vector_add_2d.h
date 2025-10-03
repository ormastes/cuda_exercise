// vector_add_2d.h - Part 17: Memory Hierarchy Evolution
#ifndef VECTOR_ADD_2D_H
#define VECTOR_ADD_2D_H

#include <cuda_runtime.h>

// Device helper function
__device__ inline float square(float x) {
    return x * x;
}

// ============================================================================
// Basic Kernels
// ============================================================================

// Basic 2D vector addition
__global__ void vector_add_2d(const float* A, const float* B, float* C,
                               int width, int height);

// ============================================================================
// Memory Hierarchy Optimized Kernels (Part 17)
// ============================================================================

// Shared memory version
__global__ void vector_add_2d_shared(const float* A, const float* B, float* C,
                                      int width, int height);

// Coalesced access pattern
__global__ void vector_add_2d_coalesced(const float* A, const float* B, float* C,
                                         int width, int height);

// Constant memory version
extern __constant__ float c_scalar[1];
__global__ void vector_add_2d_constant(const float* A, const float* B, float* C,
                                        int width, int height);

// Register-optimized with unrolling
__global__ void vector_add_2d_unrolled(const float* A, const float* B, float* C,
                                        int width, int height);

// Bank conflict-free shared memory
__global__ void vector_add_2d_bank_conflict_free(const float* A, const float* B, float* C,
                                                   int width, int height);

// Strided access (for comparison)
__global__ void vector_add_2d_strided(const float* A, const float* B, float* C,
                                       int width, int height, int stride);

// ============================================================================
// Reduction Operations
// ============================================================================

__global__ void reduce_sum(const float* input, float* output, int N, int stride);

// ============================================================================
// Debugging Kernels
// ============================================================================

__global__ void vector_add_2d_with_bug(const float* a, const float* b, float* c,
                                        int width, int height);

__global__ void vector_add_2d_with_assertion(const float* a, const float* b, float* c,
                                               int width, int height);

#endif // VECTOR_ADD_2D_H
