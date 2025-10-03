// vector_add_memory.cu - Demonstrates memory hierarchy effects on performance
#include <cuda_runtime.h>
#include <stdio.h>

// Global memory version - baseline
__global__ void vector_add_global(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // All accesses to global memory
    }
}

// Shared memory version - demonstrates caching
__global__ void vector_add_shared(const float* A, const float* B, float* C, int N) {
    extern __shared__ float sdata[];
    float* sA = sdata;
    float* sB = &sdata[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    if (idx < N) {
        sA[tid] = A[idx];
        sB[tid] = B[idx];
    }
    __syncthreads();

    // Compute from shared memory
    if (idx < N) {
        C[idx] = sA[tid] + sB[tid];
    }
}

// Constant memory demonstration
__constant__ float const_scalar[1];

__global__ void vector_add_constant_scalar(const float* A, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // const_scalar is broadcast efficiently to all threads
        C[idx] = A[idx] + const_scalar[0];
    }
}

// Texture memory version using texture objects (modern API)
// Note: Texture memory is best for spatially-coherent read patterns
__global__ void vector_add_texture(cudaTextureObject_t tex_A, cudaTextureObject_t tex_B,
                                  float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Texture fetches with automatic caching
        float a = tex1Dfetch<float>(tex_A, idx);
        float b = tex1Dfetch<float>(tex_B, idx);
        C[idx] = a + b;
    }
}

// Register optimization - unrolled version
__global__ void vector_add_unrolled(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // Process 4 elements per thread using registers
    float a0, a1, a2, a3;
    float b0, b1, b2, b3;

    if (idx < N) {
        a0 = A[idx];
        b0 = B[idx];
    }
    if (idx + blockDim.x < N) {
        a1 = A[idx + blockDim.x];
        b1 = B[idx + blockDim.x];
    }
    if (idx + 2*blockDim.x < N) {
        a2 = A[idx + 2*blockDim.x];
        b2 = B[idx + 2*blockDim.x];
    }
    if (idx + 3*blockDim.x < N) {
        a3 = A[idx + 3*blockDim.x];
        b3 = B[idx + 3*blockDim.x];
    }

    // All computation in registers
    if (idx < N) C[idx] = a0 + b0;
    if (idx + blockDim.x < N) C[idx + blockDim.x] = a1 + b1;
    if (idx + 2*blockDim.x < N) C[idx + 2*blockDim.x] = a2 + b2;
    if (idx + 3*blockDim.x < N) C[idx + 3*blockDim.x] = a3 + b3;
}

// Coalesced vs strided access demonstration
__global__ void vector_add_strided(const float* A, const float* B, float* C, int N, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < N) {
        // Strided access pattern - poor memory coalescing
        C[idx] = A[idx] + B[idx];
    }
}

// Bank conflict demonstration with shared memory
__global__ void vector_add_bank_conflict(const float* A, const float* B, float* C, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Intentional bank conflict: threads access with stride of 32
        // causing all threads in a warp to access the same bank
        sdata[tid * 32 % 256] = A[idx];
        __syncthreads();

        float val = sdata[tid * 32 % 256] + B[idx];
        C[idx] = val;
    }
}

// Optimized bank conflict-free version
__global__ void vector_add_bank_conflict_free(const float* A, const float* B, float* C, int N) {
    __shared__ float sdata[256 + 8];  // Padding to avoid bank conflicts

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Conflict-free access pattern
        sdata[tid] = A[idx];
        __syncthreads();

        float val = sdata[tid] + B[idx];
        C[idx] = val;
    }
}