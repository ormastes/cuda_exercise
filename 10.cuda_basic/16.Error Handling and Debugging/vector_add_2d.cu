// vector_add_2d.cu - 2D vector operations implementation
#include "vector_add_2d.h"
#include <cstdio>
#include <cassert>
#include <cmath>

// Simple 2D vector addition kernel
__global__ void vector_add_2d(const float* a, const float* b, float* c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        c[idx] = a[idx] + b[idx];
    }
}

// 2D reduction sum kernel (sum all elements)
__global__ void reduce_sum_2d(const float* input, float* output, int width, int height) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data into shared memory
    float value = 0.0f;
    if (x < width && y < height) {
        int idx = y * width + x;
        value = input[idx];
    }
    sdata[tid] = value;
    __syncthreads();

    // Reduction in shared memory
    int numThreads = blockDim.x * blockDim.y;
    for (int s = numThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Kernel with intentional bug (missing boundary check)
__global__ void vector_add_2d_with_bug(const float* a, const float* b, float* c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // BUG: Missing boundary check - will cause out-of-bounds access
    int idx = y * width + x;
    c[idx] = a[idx] + b[idx];
}

// Kernel with assertion for debugging
__global__ void vector_add_2d_with_assert(const float* a, const float* b, float* c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // Assert that input values are valid (not NaN)
        assert(!isnan(a[idx]) && !isnan(b[idx]));
        c[idx] = a[idx] + b[idx];
        // Assert that output is valid
        assert(!isnan(c[idx]));
    }
}