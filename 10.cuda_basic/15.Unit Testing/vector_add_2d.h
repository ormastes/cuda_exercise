// vector_add_2d.h - Header for 2D vector operations
#pragma once
#include <cuda_runtime.h>

// Simple helper device function for testing
__device__ inline float compute_sum(float a, float b) {
    return a + b;
}

// Kernel declarations
__global__ void vector_add_2d(const float* a, const float* b, float* c, int width, int height);
__global__ void reduce_sum_2d(const float* input, float* output, int width, int height);