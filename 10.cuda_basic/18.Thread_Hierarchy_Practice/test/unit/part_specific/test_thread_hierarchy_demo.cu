/**
 * Unit tests extracted from thread_hierarchy_demo
 * Tests comprehensive thread hierarchy concepts
 */

#include <gtest/gtest.h>
#include "gpu_gtest.h"
#include "cuda_utils.h"
#define BUILDING_LIB
#include "kernels/matrix_multiply.cu"
#include "part_specific/thread_operations.cu"

class ThreadHierarchyDemoTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default matrix size for testing
        N = 256;  // Use smaller size for faster tests
        size = N * N * sizeof(float);

        // Allocate host memory
        h_A = (float*)malloc(size);
        h_B = (float*)malloc(size);
        h_C_ref = (float*)malloc(size);
        h_C_test = (float*)malloc(size);

        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(rand() % 10) / 10.0f;
            h_B[i] = (float)(rand() % 10) / 10.0f;
        }

        // Allocate device memory
        d_A = cuda_malloc<float>(size);
        d_B = cuda_malloc<float>(size);
        d_C = cuda_malloc<float>(size);

        cuda_memcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cuda_memcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Compute CPU reference
        computeReference();
    }

    void TearDown() override {
        free(h_A);
        free(h_B);
        free(h_C_ref);
        free(h_C_test);
        cuda_free(d_A);
        cuda_free(d_B);
        cuda_free(d_C);
    }

    void computeReference() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += h_A[i * N + k] * h_B[k * N + j];
                }
                h_C_ref[i * N + j] = sum;
            }
        }
    }

    bool verifyResults(float tolerance = 1e-3f) {
        for (int i = 0; i < N * N; i++) {
            if (fabs(h_C_ref[i] - h_C_test[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    template<typename KernelFunc>
    float benchmarkKernel(KernelFunc kernel, dim3 grid, dim3 block, int iterations = 10) {
        // Warmup
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < iterations; i++) {
            kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        cudaDeviceSynchronize();
        timer.stop();

        return timer.elapsed_ms() / iterations;
    }

    int N;
    size_t size;
    float *h_A, *h_B, *h_C_ref, *h_C_test;
    float *d_A, *d_B, *d_C;
};

// Test performance progression from Part 17 to Part 18
GPU_TEST_F(ThreadHierarchyDemoTest, PerformanceProgression) {
    float gflops_factor = (2.0f * N * N * N) / 1e9f;
    float baseline_time = 0;

    // Test 1: Naive implementation (Part 17 baseline)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        float time = benchmarkKernel(matmul_naive, grid, block);
        baseline_time = time;
        float gflops = gflops_factor / (time / 1000.0f);

        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults());

        printf("Part 17 Naive: %.3f ms, %.2f GFLOPS\n", time, gflops);
    }

    // Test 2: Basic tiled (Part 17)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        float time = benchmarkKernel(matmul_basic_tiled, grid, block);
        float speedup = baseline_time / time;
        float gflops = gflops_factor / (time / 1000.0f);

        matmul_basic_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults());

        printf("Part 17 Basic Tiled: %.3f ms, %.2f GFLOPS, %.2fx speedup\n",
               time, gflops, speedup);

        // Expect at least 2x speedup from naive
        EXPECT_GT(speedup, 2.0f);
    }

    // Test 3: Optimized tiled (Part 18)
    {
        const int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        float time = benchmarkKernel(matmul_tiled<32>, grid, block);
        float speedup = baseline_time / time;
        float gflops = gflops_factor / (time / 1000.0f);

        matmul_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults());

        printf("Part 18 Optimized Tiled: %.3f ms, %.2f GFLOPS, %.2fx speedup\n",
               time, gflops, speedup);

        // Expect further improvement
        EXPECT_GT(speedup, 3.0f);
    }

    // Test 4: Thread coarsened (Part 18)
    {
        const int TILE = 16;
        const int COARSE = 2;
        dim3 block(TILE / COARSE, TILE / COARSE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        float time = benchmarkKernel(matmul_coarsened<16, 2>, grid, block);
        float speedup = baseline_time / time;
        float gflops = gflops_factor / (time / 1000.0f);

        matmul_coarsened<16, 2><<<grid, block>>>(d_A, d_B, d_C, N);
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults(1e-2f)); // Higher tolerance for coarsening

        printf("Part 18 Thread Coarsened: %.3f ms, %.2f GFLOPS, %.2fx speedup\n",
               time, gflops, speedup);
    }

    // Test 5: Warp optimized (Part 18)
    {
        const int WARP_SIZE = 32;
        const int WARPS_PER_BLOCK = 8;
        dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
        int gridX = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
        int gridY = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
        dim3 grid(gridX, gridY);

        float time = benchmarkKernel(matmul_warp_opt, grid, block);
        float speedup = baseline_time / time;
        float gflops = gflops_factor / (time / 1000.0f);

        matmul_warp_opt<<<grid, block>>>(d_A, d_B, d_C, N);
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults());

        printf("Part 18 Warp Optimized: %.3f ms, %.2f GFLOPS, %.2fx speedup\n",
               time, gflops, speedup);
    }
}

// Test different thread configurations
GPU_TEST_F(ThreadHierarchyDemoTest, ThreadConfigurationAnalysis) {
    int blockSizes[] = {8, 16, 32};
    float gflops_factor = (2.0f * N * N * N) / 1e9f;

    printf("\n=== Thread Configuration Analysis ===\n");
    printf("Block Size | Time (ms) | GFLOPS | Occupancy\n");
    printf("-----------|-----------|--------|----------\n");

    for (int i = 0; i < 3; i++) {
        int blockSize = blockSizes[i];
        dim3 block(blockSize, blockSize);
        dim3 grid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

        // Calculate occupancy
        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            blockSize == 16 ? matmul_tiled<16> :
            blockSize == 32 ? matmul_tiled<32> :
            matmul_tiled<16>,
            blockSize * blockSize,
            0
        );

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        float occupancy = (float)(maxActiveBlocks * blockSize * blockSize) /
                         prop.maxThreadsPerMultiProcessor;

        // Benchmark
        float time;
        if (blockSize == 16) {
            time = benchmarkKernel(matmul_tiled<16>, grid, block);
        } else if (blockSize == 32) {
            time = benchmarkKernel(matmul_tiled<32>, grid, block);
        } else {
            time = benchmarkKernel(matmul_tiled<16>, grid, block);
        }

        float gflops = gflops_factor / (time / 1000.0f);

        printf("%9dx%d | %9.3f | %6.2f | %7.1f%%\n",
               blockSize, blockSize, time, gflops, occupancy * 100);

        // Verify correctness
        if (blockSize == 16) {
            matmul_tiled<16><<<grid, block>>>(d_A, d_B, d_C, N);
        } else if (blockSize == 32) {
            matmul_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
        } else {
            matmul_tiled<16><<<grid, block>>>(d_A, d_B, d_C, N);
        }
        cuda_memcpy(h_C_test, d_C, size, cuda_memcpyDeviceToHost);
        EXPECT_TRUE(verifyResults());
    }
}

// Test with different matrix sizes
class MatrixSizeTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        N = GetParam();
        size = N * N * sizeof(float);

        h_A = (float*)malloc(size);
        h_B = (float*)malloc(size);
        h_C = (float*)malloc(size);

        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(rand() % 10) / 10.0f;
            h_B[i] = (float)(rand() % 10) / 10.0f;
        }

        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cuda_memcpy(d_A, h_A, size, cuda_memcpyHostToDevice);
        cuda_memcpy(d_B, h_B, size, cuda_memcpyHostToDevice);
    }

    void TearDown() override {
        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    int N;
    size_t size;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
};

GPU_TEST_P(MatrixSizeTest, PerformanceScaling) {
    const int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    CudaTimer timer;
    timer.start();
    matmul_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    timer.stop();

    float gflops = ((2.0f * N * N * N) / 1e9f) / (timer.elapsed_ms() / 1000.0f);

    printf("N=%4d: %.3f ms, %.2f GFLOPS\n", N, timer.elapsed_ms(), gflops);

    // Performance should scale well
    EXPECT_GT(gflops, 10.0f); // Expect at least 10 GFLOPS
}

INSTANTIATE_TEST_SUITE_P(
    MatrixSizes,
    MatrixSizeTest,
    ::testing::Values(128, 256, 512, 1024)
);

// Test device capabilities
TEST(DeviceCapabilities, ThreadHierarchyLimits) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n=== Device Thread Hierarchy Capabilities ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max grid dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max block dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    // Verify device meets minimum requirements
    EXPECT_GE(prop.maxThreadsPerBlock, 512);
    EXPECT_EQ(prop.warpSize, 32);
    EXPECT_GE(prop.sharedMemPerBlock, 16384); // At least 16KB
}