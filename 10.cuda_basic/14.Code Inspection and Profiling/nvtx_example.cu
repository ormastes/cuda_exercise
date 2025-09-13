// nvtx_example.cu - Demonstrates NVTX profiling markers
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Simple kernel for matrix multiplication
__global__ void matrixMulKernel(float* C, const float* A, const float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Vector addition kernel
__global__ void vectorAddKernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void initializeData(float* data, int size, float value) {
    for (int i = 0; i < size; i++) {
        data[i] = value + (i % 100) * 0.01f;
    }
}

int main() {
    const int N = 512;  // Matrix/vector size
    const int MATRIX_SIZE = N * N;
    const int VECTOR_SIZE = N * 1024;

    // Create main NVTX range for the entire program
    nvtxRangePushA("Main Program");

    // Memory allocation with NVTX markers
    nvtxRangePushA("Memory Allocation");
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices
    float *h_vec_a, *h_vec_b, *h_vec_c;  // Host vectors
    float *d_vec_a, *d_vec_b, *d_vec_c;  // Device vectors

    // Allocate host memory
    h_A = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_B = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_C = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_vec_a = (float*)malloc(VECTOR_SIZE * sizeof(float));
    h_vec_b = (float*)malloc(VECTOR_SIZE * sizeof(float));
    h_vec_c = (float*)malloc(VECTOR_SIZE * sizeof(float));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_a, VECTOR_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_b, VECTOR_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_c, VECTOR_SIZE * sizeof(float)));
    nvtxRangePop();  // End Memory Allocation

    // Initialize data with NVTX marker
    nvtxRangePushA("Data Initialization");
    initializeData(h_A, MATRIX_SIZE, 1.0f);
    initializeData(h_B, MATRIX_SIZE, 2.0f);
    initializeData(h_vec_a, VECTOR_SIZE, 1.0f);
    initializeData(h_vec_b, VECTOR_SIZE, 2.0f);
    nvtxRangePop();  // End Data Initialization

    // Copy data to device with colored NVTX marker
    nvtxEventAttributes_t copyAttrib = {0};
    copyAttrib.version = NVTX_VERSION;
    copyAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    copyAttrib.colorType = NVTX_COLOR_ARGB;
    copyAttrib.color = 0xFFFF0000;  // Red
    copyAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    copyAttrib.message.ascii = "Host to Device Copy";

    nvtxRangePushEx(&copyAttrib);
    CHECK_CUDA(cudaMemcpy(d_A, h_A, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vec_a, h_vec_a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vec_b, h_vec_b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();  // End Host to Device Copy

    // Matrix multiplication with NVTX marker
    nvtxEventAttributes_t matrixAttrib = {0};
    matrixAttrib.version = NVTX_VERSION;
    matrixAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    matrixAttrib.colorType = NVTX_COLOR_ARGB;
    matrixAttrib.color = 0xFF00FF00;  // Green
    matrixAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    matrixAttrib.message.ascii = "Matrix Multiplication";

    nvtxRangePushEx(&matrixAttrib);
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMulKernel<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();  // End Matrix Multiplication

    // Vector addition with NVTX marker
    nvtxEventAttributes_t vectorAttrib = {0};
    vectorAttrib.version = NVTX_VERSION;
    vectorAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    vectorAttrib.colorType = NVTX_COLOR_ARGB;
    vectorAttrib.color = 0xFF0000FF;  // Blue
    vectorAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    vectorAttrib.message.ascii = "Vector Addition";

    nvtxRangePushEx(&vectorAttrib);
    int threadsPerBlock = 256;
    int blocksPerGrid = (VECTOR_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Multiple kernel launches to show in timeline
    for (int i = 0; i < 5; i++) {
        char iterName[32];
        sprintf(iterName, "Vector Add Iteration %d", i);
        nvtxRangePushA(iterName);
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec_c, d_vec_a, d_vec_b, VECTOR_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());
        nvtxRangePop();
    }
    nvtxRangePop();  // End Vector Addition

    // Copy results back with NVTX marker
    nvtxRangePushA("Device to Host Copy");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vec_c, d_vec_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();  // End Device to Host Copy

    // Cleanup with NVTX marker
    nvtxRangePushA("Memory Cleanup");
    free(h_A); free(h_B); free(h_C);
    free(h_vec_a); free(h_vec_b); free(h_vec_c);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_vec_a));
    CHECK_CUDA(cudaFree(d_vec_b));
    CHECK_CUDA(cudaFree(d_vec_c));
    nvtxRangePop();  // End Memory Cleanup

    nvtxRangePop();  // End Main Program

    printf("Program completed successfully!\n");
    printf("Run with: nsys profile --stats=true --trace=cuda,nvtx ./nvtx_example\n");
    return 0;
}