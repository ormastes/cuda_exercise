// vector_add_2d.cu
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__device__ float square(float x) {
    return x * x;
}

__global__ void reduceSum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug helper: volatile pointer to make shared memory visible in debugger
    volatile float* debug_sdata = sdata;

    // Load data to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Debug point: can inspect debug_sdata[tid] here
    // Uncomment for printf debugging:
    // if (blockIdx.x == 0 && tid == 0) {
    //     printf("Block 0, initial sdata[0] = %f\n", debug_sdata[0]);
    // }

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        // Force read into debuggable variable
        float result = debug_sdata[0];

        // Debug output (uncomment if needed):
        // printf("Block %d: final result = %f\n", blockIdx.x, result);

        atomicAdd(output, result);
    }
}

__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

void testVectorAdd2D() {
    int width = 1024;
    int height = 1024;
    int N = width * height;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15)/16, (height + 15)/16);
    vectorAdd2D<<<blocks, threads>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "VectorAdd2D: C[0] = " << h_C[0] << std::endl;
    std::cout << "VectorAdd2D: C[N-1] = " << h_C[N-1] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

void testReduceSum() {
    const int N = 1024 * 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float h_output = 0.0f;

    // Initialize input data
    float expected_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // Simple test: all ones
        expected_sum += h_input[i];
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    // Launch kernel with dynamic shared memory
    size_t sharedMemSize = blockSize * sizeof(float);
    reduceSum<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Check result
    std::cout << "ReduceSum: Expected sum = " << expected_sum << std::endl;
    std::cout << "ReduceSum: Computed sum = " << h_output << std::endl;
    std::cout << "ReduceSum: Error = " << std::abs(h_output - expected_sum) << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
}

int main() {
    // Check CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Run tests
    std::cout << "=== Testing VectorAdd2D ===" << std::endl;
    testVectorAdd2D();
    std::cout << std::endl;

    std::cout << "=== Testing ReduceSum ===" << std::endl;
    testReduceSum();

    return 0;
}
