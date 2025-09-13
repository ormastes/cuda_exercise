// memory_errors.cu - Example with intentional errors for sanitizer testing
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

__global__ void buggyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Error 1: Out-of-bounds access
    if (idx <= n) {  // BUG: Should be idx < n
        data[idx] = idx * 2.0f;
    }

    // Error 2: Race condition in shared memory
    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[idx];
    // BUG: Missing __syncthreads() here!

    // Use shared memory (will have race condition)
    if (threadIdx.x < 255) {
        data[idx] = sdata[threadIdx.x] + sdata[threadIdx.x + 1];
    }

    // Error 3: Conditional syncthreads (potential deadlock)
    if (threadIdx.x < 128) {
        __syncthreads();  // BUG: Not all threads reach this
    }
}

int main() {
    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    size_t size = N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_data, size));

    // Launch kernel with bugs
    buggyKernel<<<numBlocks, blockSize>>>(d_data, N);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_data));

    printf("Program completed (but may have errors!)\n");
    printf("Run with compute-sanitizer to detect issues.\n");
    return 0;
}