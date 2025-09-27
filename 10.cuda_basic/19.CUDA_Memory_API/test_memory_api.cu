// test_memory_api.cu - Unit tests for CUDA Memory API operations
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <chrono>
#include <cmath>

// Test fixture for memory API tests
class CudaMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is initialized
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA initialization failed: " << cudaGetErrorString(err);
        }
    }

    void TearDown() override {
        // Clean up any remaining allocations
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during test: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceReset();
    }
};

// Simple kernel for testing
__global__ void test_kernel(float* data, int N, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = data[tid] * value;
    }
}

// Test basic allocation and deallocation
TEST_F(CudaMemoryTest, BasicAllocation) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);
    ASSERT_NE(d_data, nullptr);

    // Verify memory is accessible
    ASSERT_EQ(cudaMemset(d_data, 0, bytes), cudaSuccess);

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test memory copy operations
TEST_F(CudaMemoryTest, MemoryCopy) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Allocate host and device memory
    std::unique_ptr<float[]> h_data(new float[N]);
    float* d_data = nullptr;

    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Copy to device
    ASSERT_EQ(cudaMemcpy(d_data, h_data.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    test_kernel<<<gridSize, blockSize>>>(d_data, N, 2.0f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy back
    std::unique_ptr<float[]> h_result(new float[N]);
    ASSERT_EQ(cudaMemcpy(h_result.get(), d_data, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i] * 2.0f);
    }

    cudaFree(d_data);
}

// Test pinned memory
TEST_F(CudaMemoryTest, PinnedMemory) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float* h_pinned = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault), cudaSuccess);
    ASSERT_NE(h_pinned, nullptr);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_pinned[i] = (float)i;
    }

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Transfer should work
    ASSERT_EQ(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify data integrity
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_pinned[i], (float)i);
    }

    ASSERT_EQ(cudaFreeHost(h_pinned), cudaSuccess);
    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test unified memory
TEST_F(CudaMemoryTest, UnifiedMemory) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float* unified_data = nullptr;
    ASSERT_EQ(cudaMallocManaged(&unified_data, bytes), cudaSuccess);
    ASSERT_NE(unified_data, nullptr);

    // Initialize on host
    for (int i = 0; i < N; i++) {
        unified_data[i] = (float)i;
    }

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    test_kernel<<<gridSize, blockSize>>>(unified_data, N, 3.0f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Verify on host
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(unified_data[i], (float)i * 3.0f);
    }

    ASSERT_EQ(cudaFree(unified_data), cudaSuccess);
}

// Test zero-copy memory
TEST_F(CudaMemoryTest, ZeroCopyMemory) {
    // Check if device supports mapped memory
    cudaDeviceProp prop;
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

    if (!prop.canMapHostMemory) {
        GTEST_SKIP() << "Device does not support mapped memory";
    }

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Enable mapped memory
    ASSERT_EQ(cudaSetDeviceFlags(cudaDeviceMapHost), cudaSuccess);

    float* h_data = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_data, bytes, cudaHostAllocMapped), cudaSuccess);
    ASSERT_NE(h_data, nullptr);

    // Get device pointer
    float* d_data = nullptr;
    ASSERT_EQ(cudaHostGetDevicePointer(&d_data, h_data, 0), cudaSuccess);
    ASSERT_NE(d_data, nullptr);

    // Initialize on host
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Launch kernel with device pointer
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    test_kernel<<<gridSize, blockSize>>>(d_data, N, 4.0f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Verify on host
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_data[i], (float)i * 4.0f);
    }

    ASSERT_EQ(cudaFreeHost(h_data), cudaSuccess);
}

// Test memory info query
TEST_F(CudaMemoryTest, MemoryInfo) {
    size_t free_mem, total_mem;
    ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);

    EXPECT_GT(free_mem, 0);
    EXPECT_GT(total_mem, 0);
    EXPECT_LE(free_mem, total_mem);

    // Allocate some memory and check again
    const size_t alloc_size = 100 * 1024 * 1024; // 100 MB
    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, alloc_size), cudaSuccess);

    size_t free_mem_after, total_mem_after;
    ASSERT_EQ(cudaMemGetInfo(&free_mem_after, &total_mem_after), cudaSuccess);

    // Free memory should have decreased
    EXPECT_LT(free_mem_after, free_mem);
    EXPECT_EQ(total_mem_after, total_mem);

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test memset operations
TEST_F(CudaMemoryTest, Memset) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Set to zero
    ASSERT_EQ(cudaMemset(d_data, 0, bytes), cudaSuccess);

    // Verify
    std::unique_ptr<float[]> h_data(new float[N]);
    ASSERT_EQ(cudaMemcpy(h_data.get(), d_data, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_data[i], 0.0f);
    }

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test 2D memory operations
TEST_F(CudaMemoryTest, Memory2D) {
    const int width = 64;
    const int height = 32;
    size_t pitch;

    float* d_data = nullptr;
    ASSERT_EQ(cudaMallocPitch(&d_data, &pitch, width * sizeof(float), height), cudaSuccess);
    ASSERT_NE(d_data, nullptr);
    EXPECT_GE(pitch, width * sizeof(float));

    // Prepare host data
    std::vector<float> h_data(width * height);
    for (int i = 0; i < width * height; i++) {
        h_data[i] = (float)i;
    }

    // Copy to device
    ASSERT_EQ(cudaMemcpy2D(d_data, pitch,
                           h_data.data(), width * sizeof(float),
                           width * sizeof(float), height,
                           cudaMemcpyHostToDevice), cudaSuccess);

    // Copy back
    std::vector<float> h_result(width * height);
    ASSERT_EQ(cudaMemcpy2D(h_result.data(), width * sizeof(float),
                           d_data, pitch,
                           width * sizeof(float), height,
                           cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test async memory operations
TEST_F(CudaMemoryTest, AsyncOperations) {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // Create stream
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Allocate memory
    std::unique_ptr<float[]> h_data(new float[N]);
    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Async copy
    ASSERT_EQ(cudaMemcpyAsync(d_data, h_data.get(), bytes,
                              cudaMemcpyHostToDevice, stream), cudaSuccess);

    // Launch kernel on same stream
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    test_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, N, 5.0f);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Async copy back
    std::unique_ptr<float[]> h_result(new float[N]);
    ASSERT_EQ(cudaMemcpyAsync(h_result.get(), d_data, bytes,
                              cudaMemcpyDeviceToHost, stream), cudaSuccess);

    // Wait for stream
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i] * 5.0f);
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test memory bandwidth
TEST_F(CudaMemoryTest, MemoryBandwidth) {
    const int N = 10 * 1024 * 1024; // 10M elements
    size_t bytes = N * sizeof(float);

    // Skip if not enough memory
    size_t free_mem, total_mem;
    ASSERT_EQ(cudaMemGetInfo(&free_mem, &total_mem), cudaSuccess);
    if (free_mem < bytes * 2) {
        GTEST_SKIP() << "Not enough GPU memory for bandwidth test";
    }

    std::unique_ptr<float[]> h_data(new float[N]);
    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Measure H2D bandwidth
    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(cudaMemcpy(d_data, h_data.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    auto end = std::chrono::high_resolution_clock::now();

    float h2d_time = std::chrono::duration<float, std::milli>(end - start).count();
    float h2d_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (h2d_time / 1000.0f);

    // Measure D2H bandwidth
    start = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(cudaMemcpy(h_data.get(), d_data, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    end = std::chrono::high_resolution_clock::now();

    float d2h_time = std::chrono::duration<float, std::milli>(end - start).count();
    float d2h_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2h_time / 1000.0f);

    std::cout << "\nMemory Bandwidth Test:" << std::endl;
    std::cout << "  H2D: " << h2d_bandwidth << " GB/s" << std::endl;
    std::cout << "  D2H: " << d2h_bandwidth << " GB/s" << std::endl;

    // Basic sanity check - bandwidth should be positive
    EXPECT_GT(h2d_bandwidth, 0.0f);
    EXPECT_GT(d2h_bandwidth, 0.0f);

    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}

// Test pinned vs pageable memory performance
TEST_F(CudaMemoryTest, PinnedVsPageable) {
    const int N = 10 * 1024 * 1024; // 10M elements
    size_t bytes = N * sizeof(float);

    // Allocate pageable memory
    std::unique_ptr<float[]> h_pageable(new float[N]);

    // Allocate pinned memory
    float* h_pinned = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault), cudaSuccess);

    // Allocate device memory
    float* d_data = nullptr;
    ASSERT_EQ(cudaMalloc(&d_data, bytes), cudaSuccess);

    // Initialize both
    for (int i = 0; i < N; i++) {
        h_pageable[i] = (float)i;
        h_pinned[i] = (float)i;
    }

    // Measure pageable transfer
    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(cudaMemcpy(d_data, h_pageable.get(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    auto end = std::chrono::high_resolution_clock::now();
    float pageable_time = std::chrono::duration<float, std::milli>(end - start).count();

    // Measure pinned transfer
    start = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice), cudaSuccess);
    end = std::chrono::high_resolution_clock::now();
    float pinned_time = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "\nPinned vs Pageable Memory:" << std::endl;
    std::cout << "  Pageable: " << pageable_time << " ms" << std::endl;
    std::cout << "  Pinned: " << pinned_time << " ms" << std::endl;
    std::cout << "  Speedup: " << pageable_time / pinned_time << "x" << std::endl;

    // Pinned should generally be faster or at least not slower
    EXPECT_LE(pinned_time, pageable_time * 1.1f); // Allow 10% margin

    ASSERT_EQ(cudaFreeHost(h_pinned), cudaSuccess);
    ASSERT_EQ(cudaFree(d_data), cudaSuccess);
}