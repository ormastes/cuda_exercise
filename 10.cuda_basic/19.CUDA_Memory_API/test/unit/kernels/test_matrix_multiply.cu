/**
 * Unit tests for matrix multiplication kernels
 */

#include <gtest/gtest.h>
#include "gpu_gtest.h"

#include "kernels/matrix_multiply.cu"

class MatrixMultiplyTest : public ::testing::Test {
protected:
    void SetUp() override {
        N = 256;
        size = N * N * sizeof(float);

        h_A = (float*)malloc(size);
        h_B = (float*)malloc(size);
        h_C_ref = (float*)malloc(size);
        h_C_test = (float*)malloc(size);

        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            h_A[i] = (float)(rand() % 10) / 10.0f;
            h_B[i] = (float)(rand() % 10) / 10.0f;
        }

        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // Compute reference on CPU
        computeReference();
    }

    void TearDown() override {
        free(h_A);
        free(h_B);
        free(h_C_ref);
        free(h_C_test);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
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

    int N;
    size_t size;
    float *h_A, *h_B, *h_C_ref, *h_C_test;
    float *d_A, *d_B, *d_C;
};

GPU_TEST_F(MatrixMultiplyTest, NaiveCorrectness) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

GPU_TEST_F(MatrixMultiplyTest, BasicTiledCorrectness) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matmul_basic_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

GPU_TEST_F(MatrixMultiplyTest, OptimizedTiledCorrectness) {
    const int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matmul_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

GPU_TEST_F(MatrixMultiplyTest, CoarsenedCorrectness) {
    const int TILE = 16;
    const int COARSE = 2;
    dim3 block(TILE / COARSE, TILE / COARSE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matmul_coarsened<16, 2><<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults(1e-2f)); // Slightly higher tolerance for coarsened
}

GPU_TEST_F(MatrixMultiplyTest, WarpOptimizedCorrectness) {
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 8;
    dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    int gridX = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
    int gridY = (N + WARPS_PER_BLOCK * WARP_SIZE - 1) / (WARPS_PER_BLOCK * WARP_SIZE);
    dim3 grid(gridX, gridY);

    matmul_warp_opt<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

// Performance comparison test
TEST(MatrixMultiplyPerformance, CompareImplementations) {
    const int N = 1024;
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize with random data
    float *h_temp = (float*)malloc(size);
    for (int i = 0; i < N * N; i++) {
        h_temp[i] = (float)(rand() % 100) / 100.0f;
    }
    cudaMemcpy(d_A, h_temp, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_temp, size, cudaMemcpyHostToDevice);

    float gflops_factor = (2.0f * N * N * N) / 1e9f;

    // Test naive
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < 10; i++) {
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();

        float gflops = gflops_factor * 10 / (timer.elapsed_ms() / 1000.0f);
        printf("Naive: %.2f GFLOPS\n", gflops);
    }

    // Test optimized tiled
    {
        const int TILE = 32;
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < 10; i++) {
            matmul_tiled<32><<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();

        float gflops = gflops_factor * 10 / (timer.elapsed_ms() / 1000.0f);
        printf("Tiled: %.2f GFLOPS\n", gflops);

        // Expect significant improvement
        EXPECT_GT(gflops, 100.0f);
    }

    free(h_temp);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}