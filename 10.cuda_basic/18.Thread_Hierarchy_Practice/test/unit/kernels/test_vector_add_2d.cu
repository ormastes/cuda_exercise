/**
 * Unit tests for 2D vector addition kernels
 */

#include <gtest/gtest.h>
#include "gpu_gtest.h"

#include "kernels/vector_add_2d.cu"

class VectorAdd2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test dimensions
        width = 1024;
        height = 768;
        size = width * height;
        bytes = size * sizeof(float);

        // Allocate host memory
        h_a = (float*)malloc(bytes);
        h_b = (float*)malloc(bytes);
        h_c = (float*)malloc(bytes);
        h_ref = (float*)malloc(bytes);

        // Initialize with test data
        for (int i = 0; i < size; i++) {
            h_a[i] = (float)i;
            h_b[i] = (float)(i * 2);
            h_ref[i] = h_a[i] + h_b[i];  // Reference result
        }

        // Allocate device memory
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        // Copy input data to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        free(h_a);
        free(h_b);
        free(h_c);
        free(h_ref);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    bool verifyResults(float tolerance = 1e-5f) {
        for (int i = 0; i < size; i++) {
            if (fabs(h_c[i] - h_ref[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    int width, height, size;
    size_t bytes;
    float *h_a, *h_b, *h_c, *h_ref;
    float *d_a, *d_b, *d_c;
};

// Test basic 2D vector addition
GPU_TEST_F(VectorAdd2DTest, Basic2D) {
    dim3 block(16, 16);
    dim3 grid = calculate_grid_2d(width, height, block);

    vector_add_2d_basic<<<grid, block>>>(d_a, d_b, d_c, width, height);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

// Test coalesced 2D vector addition
GPU_TEST_F(VectorAdd2DTest, Coalesced2D) {
    dim3 block(32, 8);
    dim3 grid = calculate_grid_2d(width/4, height, block);  // Each thread handles 4 elements
    int elements_per_thread = 4;

    vector_add_2d_coalesced<<<grid, block>>>(
        d_a, d_b, d_c, width, height, elements_per_thread);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

// Test grid-stride 2D vector addition
GPU_TEST_F(VectorAdd2DTest, GridStride2D) {
    // Use smaller grid to test grid-stride functionality
    dim3 block(16, 16);
    dim3 grid(8, 8);  // Intentionally small grid

    vector_add_2d_grid_stride<<<grid, block>>>(d_a, d_b, d_c, width, height);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

// Test tiled 2D vector addition
GPU_TEST_F(VectorAdd2DTest, Tiled2D) {
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;

    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid = calculate_grid_2d(width, height, block);

    vector_add_2d_tiled<TILE_WIDTH, TILE_HEIGHT><<<grid, block>>>(
        d_a, d_b, d_c, width, height);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(verifyResults());
}

// Test with non-power-of-2 dimensions
GPU_TEST_F(VectorAdd2DTest, NonPowerOfTwo) {
    // Use odd dimensions
    int odd_width = 1023;
    int odd_height = 767;
    int odd_size = odd_width * odd_height;
    size_t odd_bytes = odd_size * sizeof(float);

    // Allocate and initialize
    float *h_odd_a = (float*)malloc(odd_bytes);
    float *h_odd_b = (float*)malloc(odd_bytes);
    float *h_odd_c = (float*)malloc(odd_bytes);

    for (int i = 0; i < odd_size; i++) {
        h_odd_a[i] = (float)i;
        h_odd_b[i] = (float)(i * 2);
    }

    float *d_odd_a, *d_odd_b, *d_odd_c;
    cudaMalloc(&d_odd_a, odd_bytes);
    cudaMalloc(&d_odd_b, odd_bytes);
    cudaMalloc(&d_odd_c, odd_bytes);

    cudaMemcpy(d_odd_a, h_odd_a, odd_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_odd_b, h_odd_b, odd_bytes, cudaMemcpyHostToDevice);

    // Test with basic kernel
    dim3 block(16, 16);
    dim3 grid = calculate_grid_2d(odd_width, odd_height, block);

    vector_add_2d_basic<<<grid, block>>>(d_odd_a, d_odd_b, d_odd_c, odd_width, odd_height);
    CHECK_KERNEL_LAUNCH();

    cudaMemcpy(h_odd_c, d_odd_c, odd_bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool correct = verify_vector_add_2d(h_odd_a, h_odd_b, h_odd_c, odd_width, odd_height);
    EXPECT_TRUE(correct);

    // Cleanup
    free(h_odd_a);
    free(h_odd_b);
    free(h_odd_c);
    cudaFree(d_odd_a);
    cudaFree(d_odd_b);
    cudaFree(d_odd_c);
}

// Performance comparison test
TEST(VectorAdd2DPerformance, CompareConfigurations) {
    const int width = 4096;
    const int height = 4096;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    // Allocate memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize with random data
    float *h_temp = (float*)malloc(bytes);
    for (int i = 0; i < size; i++) {
        h_temp[i] = (float)(rand() % 100) / 100.0f;
    }
    cudaMemcpy(d_a, h_temp, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_temp, bytes, cudaMemcpyHostToDevice);

    printf("\n=== 2D Vector Addition Performance ===\n");
    printf("Data size: %d x %d = %d elements\n\n", width, height, size);

    // Test different configurations
    struct Config {
        const char* name;
        dim3 block;
    };

    Config configs[] = {
        {"8x8 blocks", dim3(8, 8)},
        {"16x16 blocks", dim3(16, 16)},
        {"32x8 blocks", dim3(32, 8)},
        {"8x32 blocks", dim3(8, 32)},
    };

    for (const auto& config : configs) {
        dim3 grid = calculate_grid_2d(width, height, config.block);

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < 10; i++) {
            vector_add_2d_basic<<<grid, config.block>>>(d_a, d_b, d_c, width, height);
        }
        timer.stop();

        float avg_time = timer.elapsed_ms() / 10.0f;
        float bandwidth = (3 * bytes / 1e9) / (avg_time / 1000.0f);  // GB/s

        printf("%-15s: %.3f ms, %.2f GB/s\n", config.name, avg_time, bandwidth);
    }

    // Test grid-stride with small grid
    {
        dim3 block(16, 16);
        dim3 grid(32, 32);  // Small grid for grid-stride

        CudaTimer timer;
        timer.start();
        for (int i = 0; i < 10; i++) {
            vector_add_2d_grid_stride<<<grid, block>>>(d_a, d_b, d_c, width, height);
        }
        timer.stop();

        float avg_time = timer.elapsed_ms() / 10.0f;
        float bandwidth = (3 * bytes / 1e9) / (avg_time / 1000.0f);

        printf("Grid-stride    : %.3f ms, %.2f GB/s\n", avg_time, bandwidth);
    }

    free(h_temp);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Test thread hierarchy calculations
TEST(ThreadHierarchy2D, IndexCalculations) {
    // Test various block/grid configurations
    struct TestCase {
        dim3 block;
        dim3 grid;
        int width;
        int height;
    };

    TestCase cases[] = {
        {dim3(16, 16), dim3(64, 48), 1024, 768},
        {dim3(32, 8), dim3(32, 96), 1024, 768},
        {dim3(8, 32), dim3(128, 24), 1024, 768},
    };

    for (const auto& tc : cases) {
        dim3 calculated = calculate_grid_2d(tc.width, tc.height, tc.block);

        // Grid should cover all data
        EXPECT_GE(calculated.x * tc.block.x, tc.width);
        EXPECT_GE(calculated.y * tc.block.y, tc.height);

        // Grid shouldn't be too large
        EXPECT_LT((calculated.x - 1) * tc.block.x, tc.width);
        EXPECT_LT((calculated.y - 1) * tc.block.y, tc.height);

        printf("Width=%d, Height=%d, Block=(%d,%d) -> Grid=(%d,%d)\n",
               tc.width, tc.height,
               tc.block.x, tc.block.y,
               calculated.x, calculated.y);
    }
}

// Test different tile sizes
TEST(VectorAdd2DTiled, DifferentTileSizes) {
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < size; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    printf("\n=== Tile Size Comparison ===\n");

    // Test 16x16 tiles
    {
        dim3 block(16, 16);
        dim3 grid = calculate_grid_2d(width, height, block);

        CudaTimer timer;
        timer.start();
        vector_add_2d_tiled<16, 16><<<grid, block>>>(d_a, d_b, d_c, width, height);
        timer.stop();

        printf("16x16 tiles: %.3f ms\n", timer.elapsed_ms());
    }

    // Test 32x8 tiles
    {
        dim3 block(32, 8);
        dim3 grid = calculate_grid_2d(width, height, block);

        CudaTimer timer;
        timer.start();
        vector_add_2d_tiled<32, 8><<<grid, block>>>(d_a, d_b, d_c, width, height);
        timer.stop();

        printf("32x8 tiles: %.3f ms\n", timer.elapsed_ms());
    }

    // Test 8x32 tiles
    {
        dim3 block(8, 32);
        dim3 grid = calculate_grid_2d(width, height, block);

        CudaTimer timer;
        timer.start();
        vector_add_2d_tiled<8, 32><<<grid, block>>>(d_a, d_b, d_c, width, height);
        timer.stop();

        printf("8x32 tiles: %.3f ms\n", timer.elapsed_ms());
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}