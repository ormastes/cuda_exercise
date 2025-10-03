// vector_add_2d.cu
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>  // Use NVTX3 header
#include "cuda_utils.h"  // Use our custom CUDA utilities library

__device__ float square(float x) {
    return x * x;
}

// Good memory coalescing pattern - threads access consecutive memory
__global__ void vector_add_2d_coalesced(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;  // Row-major access pattern

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// Bad memory coalescing pattern - threads access strided memory
__global__ void vector_add_2d_strided(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = x * height + y;  // Column-major access in row-major storage

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

// Optimized strided access using shared memory to improve coalescing
__global__ void vector_add_2d_strided_optimized(const float* A, const float* B, float* C,
                                                int width, int height) {
    __shared__ float tile_A[16][17];  // +1 to avoid bank conflicts
    __shared__ float tile_B[16][17];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Coalesced load into shared memory (row-major)
    if (x < width && y < height) {
        int idx_row_major = y * width + x;
        tile_A[ty][tx] = A[idx_row_major];
        tile_B[ty][tx] = B[idx_row_major];
    } else {
        tile_A[ty][tx] = 0.0f;
        tile_B[ty][tx] = 0.0f;
    }
    __syncthreads();

    // Now compute using the values from shared memory
    // This converts the strided global memory access to efficient shared memory access
    if (x < width && y < height) {
        int idx_col_major = x * height + y;  // Column-major index for output
        C[idx_col_major] = square(tile_A[ty][tx]) + tile_B[ty][tx];
    }
}

// Example with potential out-of-bounds access for debugging demos
__global__ void vector_add_2d_with_bug(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bug: Missing boundary check can cause out-of-bounds access
    int i = y * width + x;
    C[i] = square(A[i]) + B[i];  // Potential out-of-bounds when x >= width or y >= height
}

__global__ void reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Debug helper: volatile pointer to make shared memory visible in debugger
    //volatile float* debug_sdata = sdata;

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

// Optimized reduction for strided access patterns
__global__ void reduce_sum_strided(const float* input, float* output, int N, int stride) {
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

void test_vector_add_2d_coalesced() {
    nvtxRangePush("test_vector_add_2d_coalesced");

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
    vector_add_2d_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
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

void test_vector_add_2d_strided() {
    nvtxRangePush("test_vector_add_2d_strided");

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
    vector_add_2d_strided<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
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

// Performance comparison function
void compare_memory_patterns() {
    nvtxRangePush("Memory Pattern Comparison");

    int width = 1024;
    int height = 1024;
    int N = width * height;
    size_t size = N * sizeof(float);

    // Allocate memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize data
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

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up runs
    vector_add_2d_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure coalesced access
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        vector_add_2d_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coalescedTime = 0;
    CHECK_CUDA(cudaEventElapsedTime(&coalescedTime, start, stop));
    coalescedTime /= 100.0f;

    // Measure strided access
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        vector_add_2d_strided<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float stridedTime = 0;
    CHECK_CUDA(cudaEventElapsedTime(&stridedTime, start, stop));
    stridedTime /= 100.0f;

    // Measure optimized strided access
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        vector_add_2d_strided_optimized<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float stridedOptimizedTime = 0;
    CHECK_CUDA(cudaEventElapsedTime(&stridedOptimizedTime, start, stop));
    stridedOptimizedTime /= 100.0f;

    // Calculate bandwidth
    float gb_transferred = (N * sizeof(float) * 3) / (1024.0f * 1024.0f * 1024.0f); // 2 reads + 1 write

    std::cout << "\n=== Memory Pattern Performance Comparison ===" << std::endl;
    std::cout << "Array size: " << width << "x" << height << " (" << N << " elements)" << std::endl;
    std::cout << "Coalesced access time:        " << coalescedTime << " ms" << std::endl;
    std::cout << "Strided access time:          " << stridedTime << " ms" << std::endl;
    std::cout << "Strided optimized time:       " << stridedOptimizedTime << " ms" << std::endl;
    std::cout << "\nPerformance ratios:" << std::endl;
    std::cout << "Strided vs Coalesced:         " << stridedTime / coalescedTime << "x slower" << std::endl;
    std::cout << "Optimized vs Original Strided: " << stridedTime / stridedOptimizedTime << "x faster" << std::endl;
    std::cout << "Optimized vs Coalesced:        " << stridedOptimizedTime / coalescedTime << "x slower" << std::endl;
    std::cout << "\nEffective Bandwidth:" << std::endl;
    std::cout << "Coalesced:        " << gb_transferred / (coalescedTime / 1000.0f) << " GB/s" << std::endl;
    std::cout << "Strided:          " << gb_transferred / (stridedTime / 1000.0f) << " GB/s" << std::endl;
    std::cout << "Strided Optimized: " << gb_transferred / (stridedOptimizedTime / 1000.0f) << " GB/s" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    nvtxRangePop();
}

void test_memory_error() {
    nvtxRangePush("test_memory_error");

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
    vector_add_2d_with_bug<<<blocks, threads>>>(d_A, d_B, d_C, width, height);
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

void test_reduce_sum() {
    nvtxRangePush("test_reduce_sum");

    const int N = 1024 * 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    const int numBlocksOptimized = (N + blockSize * 2 - 1) / (blockSize * 2);  // For optimized version

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
    reduce_sum<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    // Test optimized version for comparison
    float h_output_optimized = 0.0f;
    float *d_output_optimized;
    CHECK_CUDA(cudaMalloc(&d_output_optimized, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_output_optimized, 0, sizeof(float)));

    nvtxRangePush("Reduce: Optimized Kernel Execution");
    reduce_sum_strided<<<numBlocksOptimized, blockSize, sharedMemSize>>>(d_input, d_output_optimized, N, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    CHECK_CUDA(cudaMemcpy(&h_output_optimized, d_output_optimized, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_output_optimized));

    // Copy result back to host
    nvtxRangePush("Reduce: D2H Copy");
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // Check result
    std::cout << "ReduceSum: Expected sum = " << expected_sum << std::endl;
    std::cout << "ReduceSum: Computed sum = " << h_output << std::endl;
    std::cout << "ReduceSum: Optimized sum = " << h_output_optimized << std::endl;
    std::cout << "ReduceSum: Error = " << std::abs(h_output - expected_sum) << std::endl;
    std::cout << "ReduceSum: Optimized Error = " << std::abs(h_output_optimized - expected_sum) << std::endl;

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
    bool runComparison = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--memory-error") {
            runMemoryError = true;
        } else if (std::string(argv[i]) == "--strided") {
            runStrided = true;
        } else if (std::string(argv[i]) == "--compare") {
            runComparison = true;
        }
    }

    // Run tests
    std::cout << "=== Testing VectorAdd2D with Coalesced Access ===" << std::endl;
    test_vector_add_2d_coalesced();
    std::cout << std::endl;

    if (runStrided) {
        std::cout << "=== Testing VectorAdd2D with Strided Access ===" << std::endl;
        std::cout << "NOTE: This will show poor memory coalescing in profiler" << std::endl;
        test_vector_add_2d_strided();
        std::cout << std::endl;
    }

    std::cout << "=== Testing ReduceSum ===" << std::endl;
    test_reduce_sum();
    std::cout << std::endl;

    if (runComparison) {
        compare_memory_patterns();
        std::cout << std::endl;
    }

    if (runMemoryError) {
        test_memory_error();
        std::cout << std::endl;
    }

    std::cout << "\nProfiling tips:" << std::endl;
    std::cout << "1. Use 'nsys profile ./vector_add_2d' to see NVTX ranges" << std::endl;
    std::cout << "2. Use 'ncu --set full ./vector_add_2d' for detailed kernel analysis" << std::endl;
    std::cout << "3. Use 'compute-sanitizer ./vector_add_2d --memory-error' to detect memory issues" << std::endl;
    std::cout << "4. Compare coalesced vs strided with './vector_add_2d --strided' in profiler" << std::endl;
    std::cout << "5. Run './vector_add_2d --compare' to see performance comparison" << std::endl;

    nvtxRangePop();
    return 0;
}
