/**
 * Unit tests for thread operation kernels
 */

#include <gtest/gtest.h>
#include "cuda_utils.h"
#include "gpu_gtest.h"
#include "part_specific/thread_operations.cu"

// Test warp divergence patterns
GPU_TEST(WarpDivergence, DivergencePatterns) {
    const int N = 1024;
    int *h_data_bad = (int*)malloc(N * sizeof(int));
    int *h_data_good = (int*)malloc(N * sizeof(int));

    int *d_data_bad = cuda_malloc<int>(N);
    int *d_data_good = cuda_malloc<int>(N);

    // Test bad divergence pattern
    divergent_kernel_bad<<<(N+255)/256, 256>>>(d_data_bad, N);
    CHECK_KERNEL_LAUNCH();

    // Test good divergence pattern
    divergent_kernel_good<<<(N+255)/256, 256>>>(d_data_good, N);
    CHECK_KERNEL_LAUNCH();

    cuda_memcpy(h_data_bad, d_data_bad, N * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_memcpy(h_data_good, d_data_good, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify both produce valid results (correctness)
    for (int i = 0; i < N; i++) {
        EXPECT_GT(h_data_bad[i], 0);
        EXPECT_GT(h_data_good[i], 0);
    }

    free(h_data_bad);
    free(h_data_good);
    cuda_free(d_data_bad);
    cuda_free(d_data_good);
}

// Test memory coalescing patterns
GPU_TEST(MemoryCoalescing, AccessPatterns) {
    const int N = 1024 * 1024;
    float *d_data;
    float *h_data = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    // Test coalesced access
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < 100; i++) {
        coalesced_access<<<(N+255)/256, 256>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    float coalesced_time = timer.elapsed_ms();

    // Test strided access (stride of 32 - very bad for coalescing)
    timer.start();
    for (int i = 0; i < 100; i++) {
        strided_access<<<32, 256>>>(d_data, 32, N);
    }
    cudaDeviceSynchronize();
    timer.stop();
    float strided_time = timer.elapsed_ms();

    // Coalesced should be significantly faster
    printf("Coalesced time: %.2f ms\n", coalesced_time);
    printf("Strided time: %.2f ms\n", strided_time);
    EXPECT_LT(coalesced_time, strided_time * 0.5f);

    // Verify correctness
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < min(256, N); i++) {
        EXPECT_NEAR(h_data[i], i * 2.0f, 0.001f);
    }

    free(h_data);
    cudaFree(d_data);
}

// Test occupancy with different register usage
GPU_TEST(Occupancy, RegisterImpact) {
    const int N = 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Test with low register usage
    high_register_kernel<8><<<(N+255)/256, 256>>>(d_data, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 1.0f * (8 * 9) / 2.0f; // Sum of 1*1, 1*2, ..., 1*8
    EXPECT_NEAR(h_data[0], expected, 0.01f);

    // Test with high register usage
    high_register_kernel<32><<<(N+255)/256, 256>>>(d_data, N);
    CHECK_KERNEL_LAUNCH();

    free(h_data);
    cudaFree(d_data);
}

// Test shared memory impact
GPU_TEST(Occupancy, SharedMemoryImpact) {
    const int N = 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Test with small shared memory
    shared_memory_kernel<<<(N+255)/256, 256, 1024>>>(d_data, N, 1024);
    CHECK_KERNEL_LAUNCH();

    // Test with large shared memory
    shared_memory_kernel<<<(N+255)/256, 256, 16384>>>(d_data, N, 16384);
    CHECK_KERNEL_LAUNCH();

    // Verify kernel ran correctly
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_GT(h_data[0], 0.0f);

    free(h_data);
    cudaFree(d_data);
}

// Test warp shuffle operations
GPU_TEST(WarpPrimitives, ShuffleSum) {
    const int N = 1024;
    const int numWarps = (N + 31) / 32;

    float *d_data, *d_output;
    float *h_data = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(numWarps * sizeof(float));

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_output, numWarps * sizeof(float));

    // Initialize with sequential values
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch warp shuffle kernel
    warp_shuffle_sum<<<(N+255)/256, 256>>>(d_data, d_output, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_output, d_output, numWarps * sizeof(float), cudaMemcpyDeviceToHost);

    // Each warp should sum to 32 (or less for last partial warp)
    for (int i = 0; i < numWarps - 1; i++) {
        EXPECT_NEAR(h_output[i], 32.0f, 0.01f);
    }

    free(h_data);
    free(h_output);
    cudaFree(d_data);
    cudaFree(d_output);
}

// Test warp vote functions
GPU_TEST(WarpPrimitives, VoteFunctions) {
    const int N = 256;
    const int numWarps = (N + 31) / 32;

    int *d_data, *d_results;
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_results = (int*)malloc(numWarps * sizeof(int));

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_results, numWarps * sizeof(int));

    // Initialize with varying values
    for (int i = 0; i < N; i++) {
        h_data[i] = i % 64; // Values 0-63
    }
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Test with threshold 30
    warp_vote_kernel<<<(N+255)/256, 256>>>(d_data, d_results, N, 30);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_results, d_results, numWarps * sizeof(int), cudaMemcpyDeviceToHost);

    // First warp (values 0-31): some pass threshold
    EXPECT_EQ(h_results[0], 0); // Some passed

    // Second warp (values 32-63): all pass threshold
    if (numWarps > 1) {
        EXPECT_EQ(h_results[1], 1); // All passed
    }

    free(h_data);
    free(h_results);
    cudaFree(d_data);
    cudaFree(d_results);
}

// Performance test for occupancy analysis
TEST(OccupancyAnalysis, CalculateOccupancy) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\n=== Occupancy Analysis ===\n");
    printf("Device: %s\n", prop.name);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

    // Test different block sizes
    int blockSizes[] = {64, 128, 256, 512, 1024};

    for (int i = 0; i < 5; i++) {
        int blockSize = blockSizes[i];

        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            high_register_kernel<16>,
            blockSize,
            0
        );

        float occupancy = (float)(maxActiveBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;
        printf("Block size %4d: occupancy = %.1f%%\n", blockSize, occupancy * 100);
    }

    // Find optimal block size
    int minGridSize, optimalBlockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        high_register_kernel<16>,
        0,
        0
    );

    printf("Optimal block size: %d\n", optimalBlockSize);
    EXPECT_GT(optimalBlockSize, 0);
}