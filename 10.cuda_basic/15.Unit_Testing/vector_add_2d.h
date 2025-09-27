// vector_add_2d.h - Header for 2D vector operations
#pragma once
#include <cuda_runtime.h>

// Kernel declarations
__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height);
__global__ void reduceSum(const float* input, float* output, int N, int stride);