#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define RADIUS 3
#define BLOCK_SIZE 256

__constant__ float c_stencil[2 * RADIUS + 1];

__global__ void stencil1D_global(float* d_in, float* d_out, int n) {
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0.0f;

    if (gindex < n) {
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int access_idx = gindex + offset;

            if (access_idx >= 0 && access_idx < n) {
                result += c_stencil[RADIUS + offset] * d_in[access_idx];
            }
        }

        d_out[gindex] = result;
    }
}

__global__ void stencil1D_shared(float* d_in, float* d_out, int n) {
    __shared__ float tile[BLOCK_SIZE + 2 * RADIUS];

    int lindex = threadIdx.x + RADIUS;
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;

    if (gindex < n) {
        tile[lindex] = d_in[gindex];
    }

    if (threadIdx.x < RADIUS) {
        int left_gindex = gindex - RADIUS;
        if (left_gindex >= 0) {
            tile[lindex - RADIUS] = d_in[left_gindex];
        } else {
            tile[lindex - RADIUS] = 0.0f;
        }

        int right_gindex = gindex + blockDim.x;
        if (right_gindex < n) {
            tile[lindex + blockDim.x] = d_in[right_gindex];
        } else {
            tile[lindex + blockDim.x] = 0.0f;
        }
    }

    __syncthreads();

    if (gindex < n) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += c_stencil[RADIUS + offset] * tile[lindex + offset];
        }
        d_out[gindex] = result;
    }
}

__global__ void stencil1D_shared_optimized(float* d_in, float* d_out, int n) {
    __shared__ float tile[BLOCK_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int gindex = blockIdx.x * blockDim.x + tid;
    int tile_start = blockIdx.x * blockDim.x - RADIUS;

    // Load main part and left halo
    int load_idx = tile_start + tid;
    if (load_idx >= 0 && load_idx < n) {
        tile[tid] = d_in[load_idx];
    } else {
        tile[tid] = 0.0f;
    }

    // Load right halo
    if (tid < 2 * RADIUS) {
        load_idx = tile_start + BLOCK_SIZE + tid;
        if (load_idx >= 0 && load_idx < n) {
            tile[BLOCK_SIZE + tid] = d_in[load_idx];
        } else {
            tile[BLOCK_SIZE + tid] = 0.0f;
        }
    }

    __syncthreads();

    if (gindex < n) {
        float result = 0.0f;
        int tile_idx = tid + RADIUS;

        #pragma unroll
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += c_stencil[RADIUS + offset] * tile[tile_idx + offset];
        }

        d_out[gindex] = result;
    }
}

void initializeStencil(float* h_stencil) {
    for (int i = 0; i < 2 * RADIUS + 1; i++) {
        h_stencil[i] = 1.0f / (2 * RADIUS + 1);
    }
}

void cpuStencil1D(float* h_in, float* h_out, float* h_stencil, int n) {
    for (int i = 0; i < n; i++) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int idx = i + offset;
            if (idx >= 0 && idx < n) {
                result += h_stencil[RADIUS + offset] * h_in[idx];
            }
        }
        h_out[i] = result;
    }
}

bool verifyResult(float* gpu_result, float* cpu_result, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n",
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int n = 1 << 20;
    const int size = n * sizeof(float);
    const int stencil_size = (2 * RADIUS + 1) * sizeof(float);

    printf("1D Stencil Computation Performance\n");
    printf("===================================\n");
    printf("Array size: %d elements\n", n);
    printf("Stencil radius: %d\n", RADIUS);
    printf("Block size: %d threads\n\n", BLOCK_SIZE);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* h_out_cpu = (float*)malloc(size);
    float* h_stencil = (float*)malloc(stencil_size);

    for (int i = 0; i < n; i++) {
        h_in[i] = (float)(i % 100) / 100.0f;
    }
    initializeStencil(h_stencil);

    float* d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_stencil, h_stencil, stencil_size);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    int numIterations = 100;

    printf("Performance Comparison:\n");
    printf("-----------------------\n");

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuStencil1D(h_in, h_out_cpu, h_stencil, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    float cpu_time = cpu_duration.count() / 1000.0f;
    printf("CPU version:                    %7.3f ms\n", cpu_time);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        stencil1D_global<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float global_time = milliseconds / numIterations;
    printf("GPU (global memory):            %7.3f ms (%.2fx speedup vs CPU)\n",
           global_time, cpu_time / global_time);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (!verifyResult(h_out, h_out_cpu, n)) {
        printf("ERROR: Global memory version produced incorrect results!\n");
    }

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        stencil1D_shared<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float shared_time = milliseconds / numIterations;
    printf("GPU (shared memory):            %7.3f ms (%.2fx speedup vs global)\n",
           shared_time, global_time / shared_time);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (!verifyResult(h_out, h_out_cpu, n)) {
        printf("ERROR: Shared memory version produced incorrect results!\n");
    }

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; i++) {
        stencil1D_shared_optimized<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float optimized_time = milliseconds / numIterations;
    printf("GPU (shared memory optimized):  %7.3f ms (%.2fx speedup vs global)\n",
           optimized_time, global_time / optimized_time);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    if (!verifyResult(h_out, h_out_cpu, n)) {
        printf("ERROR: Optimized version produced incorrect results!\n");
    } else {
        printf("\nAll GPU implementations produced correct results!\n");
    }

    printf("\nMemory Bandwidth Analysis:\n");
    printf("--------------------------\n");
    size_t bytes_accessed = 2 * n * sizeof(float);
    float bandwidth_global = bytes_accessed / (global_time * 1e6);
    float bandwidth_shared = bytes_accessed / (shared_time * 1e6);
    float bandwidth_optimized = bytes_accessed / (optimized_time * 1e6);

    printf("Global memory version:    %.2f GB/s\n", bandwidth_global);
    printf("Shared memory version:    %.2f GB/s\n", bandwidth_shared);
    printf("Optimized version:        %.2f GB/s\n", bandwidth_optimized);

    free(h_in);
    free(h_out);
    free(h_out_cpu);
    free(h_stencil);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}