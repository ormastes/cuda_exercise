#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CHUNK_SIZE (1<<18)
#define NCHUNKS 16
#define NSTREAMS 2

__global__ void process_data(float* data, int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * factor + cosf(val);
        }
        data[idx] = val;
    }
}

void check_cuda_error(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

void init_data(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(i % 1000) / 1000.0f;
    }
}

int main() {
    printf("Overlapping Transfer and Compute Example\n");
    printf("=========================================\n");
    printf("Total data size: %d chunks of %d elements\n", NCHUNKS, CHUNK_SIZE);
    printf("Number of streams: %d\n\n", NSTREAMS);

    size_t chunkBytes = CHUNK_SIZE * sizeof(float);
    size_t totalSize = NCHUNKS * CHUNK_SIZE;
    size_t totalBytes = totalSize * sizeof(float);

    float *h_data_in, *h_data_out;
    float *d_data[NSTREAMS];

    cudaMallocHost(&h_data_in, totalBytes);
    cudaMallocHost(&h_data_out, totalBytes);

    for (int i = 0; i < NSTREAMS; i++) {
        check_cuda_error(cudaMalloc(&d_data[i], chunkBytes), "cudaMalloc");
    }

    init_data(h_data_in, totalSize);

    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    int threadsPerBlock = 256;
    int blocksPerGrid = (CHUNK_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    printf("Performance Comparison:\n");
    printf("-----------------------\n");

    float* d_data_seq;
    check_cuda_error(cudaMalloc(&d_data_seq, totalBytes), "cudaMalloc sequential");

    cudaEventRecord(start);
    cudaMemcpy(d_data_seq, h_data_in, totalBytes, cudaMemcpyHostToDevice);
    for (int i = 0; i < NCHUNKS; i++) {
        process_data<<<blocksPerGrid, threadsPerBlock>>>(d_data_seq + i * CHUNK_SIZE, CHUNK_SIZE, 2.0f);
    }
    cudaMemcpy(h_data_out, d_data_seq, totalBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sequential (no overlap):     %.3f ms\n", milliseconds);
    float sequentialTime = milliseconds;

    memset(h_data_out, 0, totalBytes);
    cudaEventRecord(start);

    for (int i = 0; i < NCHUNKS; i++) {
        int streamId = i % NSTREAMS;
        int offset = i * CHUNK_SIZE;

        cudaMemcpyAsync(d_data[streamId], h_data_in + offset, chunkBytes,
                       cudaMemcpyHostToDevice, streams[streamId]);

        process_data<<<blocksPerGrid, threadsPerBlock, 0, streams[streamId]>>>(
            d_data[streamId], CHUNK_SIZE, 2.0f);

        cudaMemcpyAsync(h_data_out + offset, d_data[streamId], chunkBytes,
                       cudaMemcpyDeviceToHost, streams[streamId]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Overlapped (%d streams):      %.3f ms (%.2fx speedup)\n",
           NSTREAMS, milliseconds, sequentialTime / milliseconds);
    float overlappedTime = milliseconds;

    memset(h_data_out, 0, totalBytes);
    cudaEventRecord(start);

    for (int i = 0; i < NCHUNKS; i++) {
        int streamId = i % NSTREAMS;
        int offset = i * CHUNK_SIZE;
        cudaMemcpyAsync(d_data[streamId], h_data_in + offset, chunkBytes,
                       cudaMemcpyHostToDevice, streams[streamId]);
    }

    for (int i = 0; i < NCHUNKS; i++) {
        int streamId = i % NSTREAMS;
        process_data<<<blocksPerGrid, threadsPerBlock, 0, streams[streamId]>>>(
            d_data[streamId], CHUNK_SIZE, 2.0f);
    }

    for (int i = 0; i < NCHUNKS; i++) {
        int streamId = i % NSTREAMS;
        int offset = i * CHUNK_SIZE;
        cudaMemcpyAsync(h_data_out + offset, d_data[streamId], chunkBytes,
                       cudaMemcpyDeviceToHost, streams[streamId]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Batched operations:          %.3f ms (%.2fx speedup)\n",
           milliseconds, sequentialTime / milliseconds);

    printf("\nTransfer and Compute Analysis:\n");
    printf("-------------------------------\n");

    cudaEventRecord(start);
    for (int i = 0; i < NCHUNKS; i++) {
        cudaMemcpy(d_data_seq + i * CHUNK_SIZE, h_data_in + i * CHUNK_SIZE,
                  chunkBytes, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float h2dTime = milliseconds;
    printf("H2D Transfer only:           %.3f ms\n", h2dTime);

    cudaEventRecord(start);
    for (int i = 0; i < NCHUNKS; i++) {
        process_data<<<blocksPerGrid, threadsPerBlock>>>(
            d_data_seq + i * CHUNK_SIZE, CHUNK_SIZE, 2.0f);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float computeTime = milliseconds;
    printf("Compute only:                %.3f ms\n", computeTime);

    cudaEventRecord(start);
    for (int i = 0; i < NCHUNKS; i++) {
        cudaMemcpy(h_data_out + i * CHUNK_SIZE, d_data_seq + i * CHUNK_SIZE,
                  chunkBytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float d2hTime = milliseconds;
    printf("D2H Transfer only:           %.3f ms\n", d2hTime);

    float theoreticalMin = fmax(fmax(h2dTime, computeTime), d2hTime);
    printf("\nTheoretical minimum time:    %.3f ms\n", theoreticalMin);
    printf("Overlap efficiency:          %.1f%%\n",
           (theoreticalMin / overlappedTime) * 100);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice Capabilities:\n");
    printf("--------------------\n");
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels:          %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count:          %d\n", prop.asyncEngineCount);
    printf("Unified addressing:          %s\n",
           prop.unifiedAddressing ? "Yes" : "No");

    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_data_in);
    cudaFreeHost(h_data_out);
    for (int i = 0; i < NSTREAMS; i++) {
        cudaFree(d_data[i]);
    }
    cudaFree(d_data_seq);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}