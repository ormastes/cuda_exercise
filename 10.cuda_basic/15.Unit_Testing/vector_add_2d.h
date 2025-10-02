// vector_add_2d.h - Header for 2D vector operations
#pragma once
#include <cuda_runtime.h>

// Helper device functions
__device__ inline float square(float x) {
    return x * x;
}

__device__ inline float compute_sum(float a, float b) {
    return a + b;
}

// Kernel declarations
__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height);
__global__ void reduceSum(const float* input, float* output, int N, int stride);