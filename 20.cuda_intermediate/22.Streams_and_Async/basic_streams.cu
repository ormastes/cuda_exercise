#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (1<<20)
#define NSTREAMS 4

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

void initData(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }
}

int main() {
    printf("CUDA Streams Basic Example\n");
    printf("===========================\n");
    printf("Vector size: %d elements\n", N);
    printf("Number of streams: %d\n\n", NSTREAMS);

    size_t size = N * sizeof(float);
    int streamSize = N / NSTREAMS;
    size_t streamBytes = streamSize * sizeof(float);

    float *h_a, *h_b, *h_c, *h_c_ref;
    float *d_a, *d_b, *d_c;

    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    h_c_ref = (float*)malloc(size);

    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c");

    initData(h_a, N);
    initData(h_b, N);

    for (int i = 0; i < N; i++) {
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    int threadsPerBlock = 256;
    int blocksPerGrid = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

    printf("Execution Time Comparison:\n");
    printf("--------------------------\n");

    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    vectorAdd<<<blocksPerGrid * NSTREAMS, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sequential execution (no streams): %.3f ms\n", milliseconds);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n\n", correct ? "PASSED" : "FAILED");

    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEventRecord(start);
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Async execution (%d streams):      %.3f ms\n", NSTREAMS, milliseconds);

    correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n\n", correct ? "PASSED" : "FAILED");

    memset(h_c, 0, size);
    cudaEventRecord(start);

    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
    }
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Batched operations (%d streams):   %.3f ms\n", NSTREAMS, milliseconds);

    correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n\n", correct ? "PASSED" : "FAILED");

    printf("Stream Properties:\n");
    printf("------------------\n");
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamQuery(streams[i]);
        printf("Stream %d: Created and functional\n", i);
    }

    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    free(h_c_ref);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}