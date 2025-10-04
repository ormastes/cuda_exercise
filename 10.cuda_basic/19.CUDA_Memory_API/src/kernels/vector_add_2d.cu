/**
 * Vector Addition with 2D Grid/Block Configuration
 * Part 18: Thread Hierarchy Practice
 *
 * Demonstrates various 2D thread hierarchy configurations
 * and their impact on performance
 */

#include "vector_add_2d.h"
#include <stdio.h>
#include "cuda_utils.h"

// =============================================================================
// Basic 2D Vector Addition
// =============================================================================

/**
 * Basic 2D vector addition - maps 2D thread hierarchy to 1D data
 * Performance: Good for understanding 2D indexing
 */
__global__ void vector_add_2d_basic(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height)
{
    // Calculate 2D thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Convert to 1D index (row-major order)
    int idx = y * width + x;

    // Boundary check
    if (x < width && y < height) {
        c[idx] = a[idx] + b[idx];
    }
}

// =============================================================================
// Coalesced 2D Vector Addition
// =============================================================================

/**
 * Optimized 2D addition with coalesced memory access
 * Each thread processes multiple elements to improve efficiency
 * Performance: Better memory bandwidth utilization
 */
__global__ void vector_add_2d_coalesced(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height,
    int elements_per_thread)
{
    // Calculate starting position for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Process multiple elements per thread (coalesced access pattern)
    for (int elem = 0; elem < elements_per_thread; elem++) {
        int current_x = x + elem * gridDim.x * blockDim.x;

        if (current_x < width && y < height) {
            int idx = y * width + current_x;
            c[idx] = a[idx] + b[idx];
        }
    }
}

// =============================================================================
// Grid-Stride 2D Vector Addition
// =============================================================================

/**
 * Grid-stride loop pattern for 2D data
 * Handles arbitrary sizes efficiently
 * Performance: Excellent for large data that exceeds grid size
 */
__global__ void vector_add_2d_grid_stride(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height)
{
    // Grid dimensions
    int gridWidth = gridDim.x * blockDim.x;
    int gridHeight = gridDim.y * blockDim.y;

    // Starting position for this thread
    int startX = blockIdx.x * blockDim.x + threadIdx.x;
    int startY = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid-stride loop in 2D
    for (int y = startY; y < height; y += gridHeight) {
        for (int x = startX; x < width; x += gridWidth) {
            int idx = y * width + x;
            c[idx] = a[idx] + b[idx];
        }
    }
}

// =============================================================================
// Tiled 2D Vector Addition
// =============================================================================

/**
 * Tiled processing using shared memory
 * Demonstrates tile-based approach with 2D hierarchy
 * Performance: Good for demonstrating shared memory with 2D access
 */
template<int TILE_WIDTH, int TILE_HEIGHT>
__global__ void vector_add_2d_tiled(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height)
{
    // Shared memory tiles
    __shared__ float tile_a[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float tile_b[TILE_HEIGHT][TILE_WIDTH];

    // Thread indices within tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices
    int gx = blockIdx.x * TILE_WIDTH + tx;
    int gy = blockIdx.y * TILE_HEIGHT + ty;

    // Load data into shared memory (coalesced)
    if (gx < width && gy < height) {
        int idx = gy * width + gx;
        tile_a[ty][tx] = a[idx];
        tile_b[ty][tx] = b[idx];
    } else {
        tile_a[ty][tx] = 0.0f;
        tile_b[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute and store result
    if (gx < width && gy < height) {
        int idx = gy * width + gx;
        c[idx] = tile_a[ty][tx] + tile_b[ty][tx];
    }
}

// Explicit template instantiations
template __global__ void vector_add_2d_tiled<16, 16>(const float*, const float*, float*, int, int);
template __global__ void vector_add_2d_tiled<32, 8>(const float*, const float*, float*, int, int);
template __global__ void vector_add_2d_tiled<8, 32>(const float*, const float*, float*, int, int);

// =============================================================================
// Host Helper Functions
// =============================================================================

/**
 * Calculate optimal 2D grid dimensions for given data size
 */
dim3 calculate_grid_2d(int width, int height, dim3 block) {
    int gridX = (width + block.x - 1) / block.x;
    int gridY = (height + block.y - 1) / block.y;
    return dim3(gridX, gridY);
}

/**
 * Verify results of 2D vector addition
 */
bool verify_vector_add_2d(
    const float* a,
    const float* b,
    const float* c,
    int width,
    int height,
    float tolerance)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float expected = a[idx] + b[idx];
            if (fabs(c[idx] - expected) > tolerance) {
                printf("Mismatch at (%d, %d): expected %f, got %f\n",
                       x, y, expected, c[idx]);
                return false;
            }
        }
    }
    return true;
}