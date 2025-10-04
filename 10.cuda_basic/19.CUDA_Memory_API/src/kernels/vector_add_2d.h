/**
 * Vector Addition with 2D Grid/Block Configuration
 * Part 18: Thread Hierarchy Practice
 *
 * Header file for 2D vector addition kernels demonstrating
 * different thread hierarchy configurations
 */

#ifndef VECTOR_ADD_2D_H
#define VECTOR_ADD_2D_H

#include <cuda_runtime.h>

// Kernel declarations

/**
 * Basic 2D vector addition using 2D grid and blocks
 * Demonstrates mapping 2D thread hierarchy to 1D data
 */
__global__ void vector_add_2d_basic(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height
);

/**
 * Optimized 2D vector addition with coalesced access
 * Each thread handles multiple elements for better efficiency
 */
__global__ void vector_add_2d_coalesced(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height,
    int elements_per_thread
);

/**
 * 2D vector addition with grid-stride loop
 * Handles arbitrary data sizes efficiently
 */
__global__ void vector_add_2d_grid_stride(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height
);

/**
 * Tiled 2D vector addition using shared memory
 * Demonstrates tile-based processing with 2D hierarchy
 */
template<int TILE_WIDTH, int TILE_HEIGHT>
__global__ void vector_add_2d_tiled(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height
);

// Host helper functions

/**
 * Calculate optimal 2D grid dimensions for given data size
 */
dim3 calculate_grid_2d(int width, int height, dim3 block);

/**
 * Verify results of 2D vector addition
 */
bool verify_vector_add_2d(
    const float* a,
    const float* b,
    const float* c,
    int width,
    int height,
    float tolerance = 1e-5f
);

#endif // VECTOR_ADD_2D_H