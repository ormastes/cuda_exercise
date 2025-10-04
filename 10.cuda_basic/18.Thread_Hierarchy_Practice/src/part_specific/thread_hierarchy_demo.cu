/**
 * Thread Hierarchy Comprehensive Demo
 *
 * Part 18: Thread Hierarchy Practice
 * Purpose: Complete demonstration of thread hierarchy concepts
 *
 * This example demonstrates:
 * - Evolution from Part 17's matrix multiplication
 * - Thread hierarchy optimizations
 * - Performance comparison across different configurations
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iomanip>
#include "cuda_utils.h"

// Include our kernels
#include "kernels/matrix_multiply.cu"
#include "part_specific/thread_operations.cu"

// Helper function to initialize matrices
void initialize_matrices(float* A, float* B, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 10) / 10.0f;
        B[i] = (float)(rand() % 10) / 10.0f;
    }
}

// Helper function to verify results
bool verify_results(float* C_ref, float* C_test, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return false;
        }
    }
    return true;
}

// CPU reference implementation
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Benchmark a kernel
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, dim3 grid, dim3 block,
                       float* d_A, float* d_B, float* d_C, int N,
                       int iterations = 10) {
    // Warmup
    kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<float, std::milli>(end - start).count() / iterations;
}

void run_performance_comparison(int N) {
    printf("\n=== Performance Comparison (N=%d) ===\n", N);

    // Allocate memory
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);
    float *h_C_test = (float*)malloc(size);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize data
    initialize_matrices(h_A, h_B, N);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calculate reference on CPU (only for small sizes)
    if (N <= 256) {
        matmul_cpu(h_A, h_B, h_C_ref, N);
    }

    // Performance results
    printf("\nKernel Performance Results:\n");
    printf("%-40s | %10s | %10s | %10s\n", "Kernel", "Time (ms)", "GFLOPS", "Speedup");
    printf("%-40s-+-%10s-+-%10s-+-%10s\n",
           "----------------------------------------",
           "----------", "----------", "----------");

    float baseline_time = 0;
    float gflops_factor = (2.0f * N * N * N) / 1e9f;

    // Test 1: Naive implementation (from Part 17)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        float time = benchmark_kernel(matmul_naive, grid, block, d_A, d_B, d_C, N);
        baseline_time = time;
        float gflops = gflops_factor / (time / 1000.0f);

        printf("%-40s | %10.3f | %10.2f | %10.2fx\n",
               "Part 17: Naive", time, gflops, 1.0f);

        if (N <= 256) {
            cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
            if (verify_results(h_C_ref, h_C_test, N)) {
                printf("  ✓ Results correct\n");
            }
        }
    }

    // Test 2: Basic tiled (from Part 17)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        float time = benchmark_kernel(matmul_basic_tiled, grid, block, d_A, d_B, d_C, N);
        float gflops = gflops_factor / (time / 1000.0f);
        float speedup = baseline_time / time;

        printf("%-40s | %10.3f | %10.2f | %10.2fx\n",
               "Part 17: Basic Tiled", time, gflops, speedup);

        if (N <= 256) {
            cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
            if (verify_results(h_C_ref, h_C_test, N)) {
                printf("  ✓ Results correct\n");
            }
        }
    }

    // Test 3: Optimized tiled (Part 18 enhancement)
    {
        const int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        float time = benchmark_kernel(
            matmul_tiled<32>, grid, block, d_A, d_B, d_C, N);
        float gflops = gflops_factor / (time / 1000.0f);
        float speedup = baseline_time / time;

        printf("%-40s | %10.3f | %10.2f | %10.2fx\n",
               "Part 18: Optimized Tiled (32x32)", time, gflops, speedup);

        if (N <= 256) {
            cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
            if (verify_results(h_C_ref, h_C_test, N)) {
                printf("  ✓ Results correct\n");
            }
        }
    }

    // Test 4: Thread coarsened
    {
        const int TILE = 16;
        const int COARSE = 2;
        dim3 block(TILE / COARSE, TILE / COARSE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        float time = benchmark_kernel(
            matmul_coarsened<16, 2>, grid, block, d_A, d_B, d_C, N);
        float gflops = gflops_factor / (time / 1000.0f);
        float speedup = baseline_time / time;

        printf("%-40s | %10.3f | %10.2f | %10.2fx\n",
               "Part 18: Thread Coarsened (2x2)", time, gflops, speedup);

        if (N <= 256) {
            cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
            if (verify_results(h_C_ref, h_C_test, N)) {
                printf("  ✓ Results correct\n");
            }
        }
    }

    // Test 5: Warp optimized
    {
        const int WARP_SIZE = 32;
        const int WARPS_PER_BLOCK = 8;
        dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
        int gridX = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
        int gridY = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
        dim3 grid(gridX, gridY);

        float time = benchmark_kernel(
            matmul_warp_opt, grid, block, d_A, d_B, d_C, N);
        float gflops = gflops_factor / (time / 1000.0f);
        float speedup = baseline_time / time;

        printf("%-40s | %10.3f | %10.2f | %10.2fx\n",
               "Part 18: Warp Optimized", time, gflops, speedup);

        if (N <= 256) {
            cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
            if (verify_results(h_C_ref, h_C_test, N)) {
                printf("  ✓ Results correct\n");
            }
        }
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void analyze_thread_configurations(int N) {
    printf("\n=== Thread Configuration Analysis (N=%d) ===\n", N);

    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    printf("\nTesting different block sizes with optimized kernel:\n");
    printf("Block Size | Grid Size | Occupancy | Time (ms) | GFLOPS\n");
    printf("-----------|-----------|-----------|-----------|-------\n");

    int blockSizes[] = {8, 16, 32};
    float gflops_factor = (2.0f * N * N * N) / 1e9f;

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
            matmul_tiled<16>,  // Use 16 for 8x8
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
            time = benchmark_kernel(matmul_tiled<16>, grid, block,
                                   d_A, d_B, d_C, N, 5);
        } else if (blockSize == 32) {
            time = benchmark_kernel(matmul_tiled<32>, grid, block,
                                   d_A, d_B, d_C, N, 5);
        } else {
            time = benchmark_kernel(matmul_tiled<16>, grid, block,
                                   d_A, d_B, d_C, N, 5);
        }

        float gflops = gflops_factor / (time / 1000.0f);

        printf("%9dx%d | %9dx%d | %8.1f%% | %9.3f | %6.2f\n",
               blockSize, blockSize,
               grid.x, grid.y,
               occupancy * 100,
               time,
               gflops);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#ifndef BUILDING_LIB
int main() {
    printf("Thread Hierarchy Practice - Comprehensive Demo\n");
    printf("==============================================\n");

    // Check device
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);

    // Test different matrix sizes
    int sizes[] = {256, 512, 1024};
    for (int i = 0; i < 3; i++) {
        run_performance_comparison(sizes[i]);
        analyze_thread_configurations(sizes[i]);
    }

    printf("\n=== Key Insights ===\n");
    printf("1. Thread hierarchy optimization provides significant speedup\n");
    printf("2. Optimal tile size depends on GPU architecture\n");
    printf("3. Thread coarsening improves instruction-level parallelism\n");
    printf("4. Warp-level optimization reduces synchronization overhead\n");
    printf("5. Higher occupancy doesn't always mean better performance\n");

    printf("\n=== Evolution from Part 17 ===\n");
    printf("Part 17 focused on memory hierarchy (global, shared, registers)\n");
    printf("Part 18 adds thread hierarchy optimization:\n");
    printf("  - Optimal block/grid configuration\n");
    printf("  - Warp-level programming\n");
    printf("  - Occupancy tuning\n");
    printf("  - Thread coarsening\n");

    return 0;
}
#endif