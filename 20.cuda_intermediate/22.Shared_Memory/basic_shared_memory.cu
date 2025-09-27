#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1048576

__global__ void reverseArrayGlobal(float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[n - 1 - idx];
    }
}

__global__ void reverseArrayShared(float* d_in, float* d_out, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        int reverseIdx = n - 1 - idx;
        int reverseBlockIdx = reverseIdx / blockDim.x;
        int reverseTid = reverseIdx % blockDim.x;

        if (reverseBlockIdx == blockIdx.x) {
            sharedData[tid] = d_in[idx];
            __syncthreads();
            d_out[idx] = sharedData[blockDim.x - 1 - tid];
        } else {
            d_out[idx] = d_in[reverseIdx];
        }
    }
}

__global__ void sumReductionGlobal(float* d_in, float* d_partial, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;

    float sum = 0.0f;
    if (idx < n) {
        sum = d_in[idx];
    }

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride && idx + stride < n) {
            sum += d_in[blockStart + tid + stride];
        }
        if (tid < stride) {
            d_in[blockStart + tid] = sum;
        }
    }

    if (tid == 0) {
        d_partial[blockIdx.x] = sum;
    }
}

__global__ void sumReductionShared(float* d_in, float* d_partial, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sharedData[tid] = (idx < n) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partial[blockIdx.x] = sharedData[0];
    }
}

void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

int main() {
    float *h_in, *h_out;
    float *d_in, *d_out, *d_partial;

    size_t size = ARRAY_SIZE * sizeof(float);
    int numBlocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t partialSize = numBlocks * sizeof(float);

    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = (float)(i + 1);
    }

    checkCudaError(cudaMalloc(&d_in, size), "cudaMalloc d_in");
    checkCudaError(cudaMalloc(&d_out, size), "cudaMalloc d_out");
    checkCudaError(cudaMalloc(&d_partial, partialSize), "cudaMalloc d_partial");

    checkCudaError(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "cudaMemcpy to device");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    printf("Array Reversal Comparison (Array size: %d):\n", ARRAY_SIZE);
    printf("=============================================\n");

    cudaEventRecord(start);
    reverseArrayGlobal<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Global memory version: %.3f ms\n", milliseconds);

    cudaMemcpy(d_out, h_in, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    reverseArrayShared<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory version: %.3f ms\n", milliseconds);

    checkCudaError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    bool correct = true;
    for (int i = 0; i < ARRAY_SIZE && i < 10; i++) {
        if (h_out[i] != h_in[ARRAY_SIZE - 1 - i]) {
            correct = false;
            printf("Error at index %d: expected %.0f, got %.0f\n",
                   i, h_in[ARRAY_SIZE - 1 - i], h_out[i]);
        }
    }
    if (correct) {
        printf("Array reversal: PASSED\n");
    }

    printf("\nSum Reduction Comparison:\n");
    printf("=============================================\n");

    checkCudaError(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "cudaMemcpy to device");

    float* d_in_copy;
    checkCudaError(cudaMalloc(&d_in_copy, size), "cudaMalloc d_in_copy");
    checkCudaError(cudaMemcpy(d_in_copy, d_in, size, cudaMemcpyDeviceToDevice), "cudaMemcpy d_in_copy");

    cudaEventRecord(start);
    sumReductionGlobal<<<numBlocks, BLOCK_SIZE>>>(d_in_copy, d_partial, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Global memory reduction: %.3f ms\n", milliseconds);

    checkCudaError(cudaMemcpy(d_in_copy, d_in, size, cudaMemcpyDeviceToDevice), "cudaMemcpy restore");

    cudaEventRecord(start);
    sumReductionShared<<<numBlocks, BLOCK_SIZE>>>(d_in_copy, d_partial, ARRAY_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory reduction: %.3f ms\n", milliseconds);

    float* h_partial = (float*)malloc(partialSize);
    checkCudaError(cudaMemcpy(h_partial, d_partial, partialSize, cudaMemcpyDeviceToHost), "cudaMemcpy partial");

    float gpuSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        gpuSum += h_partial[i];
    }

    double expectedSum = (double)ARRAY_SIZE * ((double)ARRAY_SIZE + 1.0) / 2.0;
    printf("\nSum verification:\n");
    printf("Expected sum: %.0f\n", expectedSum);
    printf("GPU sum: %.0f\n", gpuSum);
    double relativeError = fabs(gpuSum - expectedSum) / expectedSum;
    if (relativeError < 0.001) {
        printf("Sum reduction: PASSED\n");
    } else {
        printf("Sum reduction: FAILED\n");
    }

    free(h_in);
    free(h_out);
    free(h_partial);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in_copy);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}