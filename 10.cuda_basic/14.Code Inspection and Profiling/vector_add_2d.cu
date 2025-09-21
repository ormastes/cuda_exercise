// vector_add_2d.cu
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>  // Use NVTX3 header
#include "cuda_utils.h"  // Use our custom CUDA utilities library

__device__ float square(float x) {
    return x * x;
}

__global__ void reduceSum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Good memory coalescing pattern - threads access consecutive memory
__global__ void vectorAdd2D_Coalesced(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;  // Row-major access pattern

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// Bad memory coalescing pattern - threads access strided memory
__global__ void vectorAdd2D_Strided(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = x * height + y;  // Column-major access in row-major storage

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// Example with potential out-of-bounds access for debugging demos
__global__ void vectorAdd2D_WithBug(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bug: Missing boundary check can cause out-of-bounds access
    int i = y * width + x;
    C[i] = square(A[i]) + B[i];  // Potential out-of-bounds when x >= width or y >= height
}

void testVectorAdd2D_Coalesced() {
    nvtxRangePush("testVectorAdd2D_Coalesced");

    int width = 1024;
    int height = 1024;
    int N = width * height;
    size_t size = N * sizeof(float);

    // NVTX marker for memory allocation
    nvtxRangePush("Host Memory Allocation");
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    nvtxRangePop();

    // Initialize data
    nvtxRangePush("Host Data Initialization");
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 1000);
        h_B[i] = static_cast<float>((2 * i) % 1000);
    }
    nvtxRangePop();

    // Device memory allocation
    nvtxRangePush("Device Memory Allocation");
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));
    nvtxRangePop();

    // Copy to device
    nvtxRangePush("Host to Device Copy");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    nvtxRangePop();

    // Kernel execution
    dim3 threads(16, 16);
    dim3 blocks((width + 15)/16, (height + 15)/16);

    nvtxRangePush("Coalesced Kernel Execution");
    vectorAdd2D_Coalesced<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    // Copy back
    nvtxRangePush("Device to Host Copy");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // Verify results
    std::cout << "Coalesced Access: C[0] = " << h_C[0] << std::endl;
    std::cout << "Coalesced Access: C[N-1] = " << h_C[N-1] << std::endl;

    // Cleanup
    nvtxRangePush("Memory Cleanup");
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    nvtxRangePop();

    nvtxRangePop();
}

void testVectorAdd2D_Strided() {
    nvtxRangePush("testVectorAdd2D_Strided");

    int width = 1024;
    int height = 1024;
    int N = width * height;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 1000);
        h_B[i] = static_cast<float>((2 * i) % 1000);
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15)/16, (height + 15)/16);

    nvtxRangePush("Strided Kernel Execution");
    vectorAdd2D_Strided<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "Strided Access: C[0] = " << h_C[0] << std::endl;
    std::cout << "Strided Access: C[N-1] = " << h_C[N-1] << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    nvtxRangePop();
}

void testMemoryError() {
    nvtxRangePush("testMemoryError");

    std::cout << "\n=== Testing Memory Error Detection ===" << std::endl;
    std::cout << "This test intentionally causes an out-of-bounds access" << std::endl;
    std::cout << "Use cuda-memcheck or compute-sanitizer to detect it" << std::endl;

    int width = 100;
    int height = 100;
    int N = width * height;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch with too many blocks - will cause out-of-bounds access
    dim3 threads(16, 16);
    dim3 blocks((width + 20)/16, (height + 20)/16);  // Intentionally wrong

    nvtxRangePush("Buggy Kernel Execution");
    vectorAdd2D_WithBug<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    cudaError_t error = cudaDeviceSynchronize();
    nvtxRangePop();

    if (error != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "Kernel completed (errors may be detected by sanitizer)" << std::endl;
    }

    // Cleanup even if kernel failed
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    nvtxRangePop();
}

void testReduceSum() {
    nvtxRangePush("testReduceSum");

    const int N = 1024 * 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    size_t size = N * sizeof(float);

    // Allocate host memory
    nvtxRangePush("Reduce: Host Allocation");
    float *h_input = (float*)malloc(size);
    float h_output = 0.0f;
    nvtxRangePop();

    // Initialize input data
    nvtxRangePush("Reduce: Data Init");
    float expected_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // Simple test: all ones
        expected_sum += h_input[i];
    }
    nvtxRangePop();

    // Allocate device memory
    nvtxRangePush("Reduce: Device Allocation");
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));
    nvtxRangePop();

    // Copy input data to device
    nvtxRangePush("Reduce: H2D Copy");
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, sizeof(float)));
    nvtxRangePop();

    // Launch kernel with dynamic shared memory
    size_t sharedMemSize = blockSize * sizeof(float);
    nvtxRangePush("Reduce: Kernel Execution");
    reduceSum<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    // Copy result back to host
    nvtxRangePush("Reduce: D2H Copy");
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // Check result
    std::cout << "ReduceSum: Expected sum = " << expected_sum << std::endl;
    std::cout << "ReduceSum: Computed sum = " << h_output << std::endl;
    std::cout << "ReduceSum: Error = " << std::abs(h_output - expected_sum) << std::endl;

    // Cleanup
    nvtxRangePush("Reduce: Cleanup");
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    nvtxRangePop();

    nvtxRangePop();
}

int main(int argc, char* argv[]) {
    nvtxRangePush("Main");

    // Check CUDA device
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    bool runMemoryError = false;
    bool runStrided = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--memory-error") {
            runMemoryError = true;
        } else if (std::string(argv[i]) == "--strided") {
            runStrided = true;
        }
    }

    // Run tests
    std::cout << "=== Testing VectorAdd2D with Coalesced Access ===" << std::endl;
    testVectorAdd2D_Coalesced();
    std::cout << std::endl;

    if (runStrided) {
        std::cout << "=== Testing VectorAdd2D with Strided Access ===" << std::endl;
        std::cout << "NOTE: This will show poor memory coalescing in profiler" << std::endl;
        testVectorAdd2D_Strided();
        std::cout << std::endl;
    }

    std::cout << "=== Testing ReduceSum ===" << std::endl;
    testReduceSum();
    std::cout << std::endl;

    if (runMemoryError) {
        testMemoryError();
        std::cout << std::endl;
    }

    std::cout << "\nProfiling tips:" << std::endl;
    std::cout << "1. Use 'nsys profile ./vector_add_2d' to see NVTX ranges" << std::endl;
    std::cout << "2. Use 'ncu --set full ./vector_add_2d' for detailed kernel analysis" << std::endl;
    std::cout << "3. Use 'compute-sanitizer ./vector_add_2d --memory-error' to detect memory issues" << std::endl;
    std::cout << "4. Compare coalesced vs strided with './vector_add_2d --strided' in profiler" << std::endl;

    nvtxRangePop();
    return 0;
}
