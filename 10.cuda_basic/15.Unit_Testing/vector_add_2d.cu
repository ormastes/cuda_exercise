// vector_add_2d.cu - 2D vector operations implementation
#include "vector_add_2d.h"
#include <cstdio>


// Simple 2D vector addition kernel
__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = x * height + y;  // Column-major access in row-major storage

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// 2D reduction sum kernel (sum all elements)
__global__ void reduceSum(const float* input, float* output, int N, int stride) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;  // Grid-stride loop

    // Coalesced load with grid-stride loop to handle strided patterns better
    float sum = 0.0f;

    // First, accumulate multiple elements per thread (coalesced reads)
    while (i < N) {
        sum += input[i];
        if (i + blockDim.x < N)
            sum += input[i + blockDim.x];  // Load two elements per thread
        i += gridDim.x * blockDim.x * 2;  // Grid-stride loop
    }

    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory (unrolled for better performance)
    if (blockDim.x >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* vdata = sdata;
        if (blockDim.x >= 64) vdata[tid] += vdata[tid + 32];
        if (blockDim.x >= 32) vdata[tid] += vdata[tid + 16];
        if (blockDim.x >= 16) vdata[tid] += vdata[tid + 8];
        if (blockDim.x >= 8) vdata[tid] += vdata[tid + 4];
        if (blockDim.x >= 4) vdata[tid] += vdata[tid + 2];
        if (blockDim.x >= 2) vdata[tid] += vdata[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}