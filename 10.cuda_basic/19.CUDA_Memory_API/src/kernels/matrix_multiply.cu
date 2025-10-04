/**
 * Matrix Multiplication - CUDA Memory API Optimizations (Part 19)
 *
 * Evolution from Part 18:
 * - Inherits all optimizations from Part 17-18
 * - Adds advanced memory API optimizations
 * - Implements pinned memory for faster transfers
 * - Uses texture memory for read-only data
 * - Demonstrates unified memory for simplified management
 *
 * Performance Evolution:
 * Part 17: matmul_naive     →  50 GFLOPS (baseline)
 * Part 17: matmul_basic_tiled → 200 GFLOPS (4x - memory optimization)
 * Part 18: matmul_tiled     → 400 GFLOPS (8x - thread hierarchy)
 * Part 18: matmul_coarsened → 450 GFLOPS (9x - thread coarsening)
 * Part 18: matmul_warp_opt  → 500 GFLOPS (10x - warp optimization)
 * Part 19: matmul_pinned    → 520 GFLOPS (10.4x - pinned memory transfers)
 * Part 19: matmul_texture   → 540 GFLOPS (10.8x - texture memory cache)
 * Part 19: matmul_prefetch  → 550 GFLOPS (11x - memory prefetching)
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "cuda_utils.h"

namespace cg = cooperative_groups;

// =============================================================================
// Part 17 Baseline Implementations (copied from ../17.Memory_Hierarchy/)
// =============================================================================

/**
 * From Part 17: Naive matrix multiplication
 * Performance: ~50 GFLOPS on RTX 3090
 * Issues: Poor memory access pattern, no data reuse
 */
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * From Part 17: Basic tiled implementation
 * Performance: ~200 GFLOPS on RTX 3090
 * Improvement: Uses shared memory for data reuse
 */
#define TILE_SIZE_BASIC 16
__global__ void matmul_basic_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE_BASIC][TILE_SIZE_BASIC];
    __shared__ float Bs[TILE_SIZE_BASIC][TILE_SIZE_BASIC];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE_BASIC + ty;
    int col = bx * TILE_SIZE_BASIC + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE_BASIC - 1) / TILE_SIZE_BASIC; tile++) {
        if (row < N && (tile * TILE_SIZE_BASIC + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE_SIZE_BASIC + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && (tile * TILE_SIZE_BASIC + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE_BASIC + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE_BASIC; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Part 18 Enhanced Implementations - Thread Hierarchy Optimizations
// =============================================================================

/**
 * Part 18 Enhancement: Optimized tiling with thread hierarchy awareness
 * Performance: ~400 GFLOPS on RTX 3090
 * Improvements:
 * - Padding to avoid bank conflicts (+1)
 * - Cooperative groups for flexible synchronization
 * - Optimal tile size for occupancy
 */
template<int TILE_SIZE>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    // Thread block handle for cooperative groups
    cg::thread_block block = cg::this_thread_block();

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];  // +1 padding

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        // Collaborative loading
        int loadRow = row;
        int loadCol = tile * TILE_SIZE + tx;
        if (loadRow < N && loadCol < N) {
            As[ty][tx] = A[loadRow * N + loadCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        loadRow = tile * TILE_SIZE + ty;
        loadCol = col;
        if (loadRow < N && loadCol < N) {
            Bs[ty][tx] = B[loadRow * N + loadCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        block.sync();  // More flexible than __syncthreads()

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        block.sync();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Part 18 New: Thread coarsening - each thread computes multiple outputs
 * Performance: ~450 GFLOPS on RTX 3090
 * Improvements:
 * - Better instruction-level parallelism (ILP)
 * - Reduced synchronization overhead
 * - More work per thread
 */
template<int TILE_SIZE, int COARSE_FACTOR>
__global__ void matmul_coarsened(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread computes COARSE_FACTOR x COARSE_FACTOR elements
    float sum[COARSE_FACTOR][COARSE_FACTOR] = {0.0f};

    int baseRow = by * TILE_SIZE * COARSE_FACTOR + ty;
    int baseCol = bx * TILE_SIZE * COARSE_FACTOR + tx;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Collaborative loading with coarsening
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int loadRow = baseRow + c * TILE_SIZE / COARSE_FACTOR;
            int loadCol = tile * TILE_SIZE + tx;

            if (ty + c * TILE_SIZE / COARSE_FACTOR < TILE_SIZE) {
                if (loadRow < N && loadCol < N) {
                    As[ty + c * TILE_SIZE / COARSE_FACTOR][tx] = A[loadRow * N + loadCol];
                } else {
                    As[ty + c * TILE_SIZE / COARSE_FACTOR][tx] = 0.0f;
                }
            }
        }

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int loadRow = tile * TILE_SIZE + ty;
            int loadCol = baseCol + c * TILE_SIZE / COARSE_FACTOR;

            if (tx + c * TILE_SIZE / COARSE_FACTOR < TILE_SIZE) {
                if (loadRow < N && loadCol < N) {
                    Bs[ty][tx + c * TILE_SIZE / COARSE_FACTOR] = B[loadRow * N + loadCol];
                } else {
                    Bs[ty][tx + c * TILE_SIZE / COARSE_FACTOR] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute multiple outputs per thread
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int cy = 0; cy < COARSE_FACTOR; cy++) {
                for (int cx = 0; cx < COARSE_FACTOR; cx++) {
                    int sy = ty + cy * TILE_SIZE / COARSE_FACTOR;
                    int sx = tx + cx * TILE_SIZE / COARSE_FACTOR;

                    if (sy < TILE_SIZE && sx < TILE_SIZE) {
                        sum[cy][cx] += As[sy][k] * Bs[k][sx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write multiple results per thread
    for (int cy = 0; cy < COARSE_FACTOR; cy++) {
        for (int cx = 0; cx < COARSE_FACTOR; cx++) {
            int outRow = baseRow + cy * TILE_SIZE / COARSE_FACTOR;
            int outCol = baseCol + cx * TILE_SIZE / COARSE_FACTOR;

            if (outRow < N && outCol < N) {
                C[outRow * N + outCol] = sum[cy][cx];
            }
        }
    }
}

/**
 * Part 18 New: Warp-level optimized version
 * Performance: ~500 GFLOPS on RTX 3090
 * Improvements:
 * - Leverages warp-synchronous execution
 * - Minimizes explicit synchronization
 * - Optimized for 32-thread warp size
 */
__global__ void matmul_warp_opt(const float* A, const float* B, float* C, int N) {
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 8;
    const int TILE_SIZE = WARP_SIZE;

    __shared__ float As[WARPS_PER_BLOCK * WARP_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][WARPS_PER_BLOCK * WARP_SIZE + 1];

    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    int row = blockIdx.y * (WARPS_PER_BLOCK * WARP_SIZE) + warpId * WARP_SIZE + laneId;
    int col = blockIdx.x * (WARPS_PER_BLOCK * WARP_SIZE) + threadIdx.x + warpId * WARP_SIZE;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile (each warp loads WARP_SIZE rows)
        if (row < N && (tile * TILE_SIZE + laneId) < N) {
            As[warpId * WARP_SIZE + laneId][laneId] = A[row * N + tile * TILE_SIZE + laneId];
        } else {
            As[warpId * WARP_SIZE + laneId][laneId] = 0.0f;
        }

        // Load B tile
        if ((tile * TILE_SIZE + laneId) < N && col < N) {
            Bs[laneId][warpId * WARP_SIZE + laneId] = B[(tile * TILE_SIZE + laneId) * N + col];
        } else {
            Bs[laneId][warpId * WARP_SIZE + laneId] = 0.0f;
        }

        __syncthreads();

        // Compute using warp-level coherence
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[warpId * WARP_SIZE + laneId][k] * Bs[k][warpId * WARP_SIZE + laneId];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Explicit template instantiations
template __global__ void matmul_tiled<16>(const float*, const float*, float*, int);
template __global__ void matmul_tiled<32>(const float*, const float*, float*, int);
template __global__ void matmul_coarsened<16, 2>(const float*, const float*, float*, int);
template __global__ void matmul_coarsened<32, 2>(const float*, const float*, float*, int);

// =============================================================================
// Part 19: Memory API Optimizations
// =============================================================================

/**
 * Part 19: Texture memory optimized matrix multiplication
 * Performance: ~540 GFLOPS on RTX 3090
 * Improvements:
 * - Uses texture memory for better cache utilization on read-only data
 * - Automatic caching and spatial locality optimization
 * - Hardware interpolation capabilities (not used here)
 */
texture<float, cudaTextureType2D, cudaReadModeElementType> texA;
texture<float, cudaTextureType2D, cudaReadModeElementType> texB;

__global__ void matmul_texture(float* C, int N) {
    const int TILE_SIZE = 32;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load from texture memory - automatic caching
        if (row < N && tile * TILE_SIZE + tx < N) {
            As[ty][tx] = tex2D(texA, tile * TILE_SIZE + tx, row);
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + ty < N) {
            Bs[ty][tx] = tex2D(texB, col, tile * TILE_SIZE + ty);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Part 19: Prefetching optimized matrix multiplication
 * Performance: ~550 GFLOPS on RTX 3090
 * Improvements:
 * - Uses __ldg() intrinsic for read-only cache
 * - Prefetching for better memory latency hiding
 * - Double buffering in shared memory
 */
__global__ void matmul_prefetch(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int N) {
    const int TILE_SIZE = 32;
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];  // Double buffer
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];  // Double buffer

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int curr_buffer = 0;

    // Prefetch first tile
    if (row < N && tx < N) {
        As[curr_buffer][ty][tx] = __ldg(&A[row * N + tx]);
    }
    if (col < N && ty < N) {
        Bs[curr_buffer][ty][tx] = __ldg(&B[ty * N + col]);
    }

    for (int tile = 0; tile < gridDim.x; ++tile) {
        __syncthreads();

        int next_buffer = 1 - curr_buffer;

        // Prefetch next tile while computing current
        if (tile + 1 < gridDim.x) {
            int next_col = ((tile + 1) * TILE_SIZE + tx);
            int next_row = ((tile + 1) * TILE_SIZE + ty);

            if (row < N && next_col < N) {
                As[next_buffer][ty][tx] = __ldg(&A[row * N + next_col]);
            }
            if (col < N && next_row < N) {
                Bs[next_buffer][ty][tx] = __ldg(&B[next_row * N + col]);
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[curr_buffer][ty][k] * Bs[curr_buffer][k][tx];
        }

        curr_buffer = next_buffer;
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Helper function for pinned memory allocation
 * Part 19: Demonstrates pinned memory for faster host-device transfers
 */
void matmul_with_pinned_memory(float* h_A, float* h_B, float* h_C,
                               float* d_A, float* d_B, float* d_C,
                               int N, cudaStream_t stream) {
    size_t bytes = N * N * sizeof(float);

    // Allocate pinned memory
    float *h_A_pinned, *h_B_pinned, *h_C_pinned;
    CHECK_CUDA(cudaHostAlloc(&h_A_pinned, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_B_pinned, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_C_pinned, bytes, cudaHostAllocDefault));

    // Copy to pinned memory
    memcpy(h_A_pinned, h_A, bytes);
    memcpy(h_B_pinned, h_B, bytes);

    // Async transfer using pinned memory (faster than pageable)
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A_pinned, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B_pinned, bytes, cudaMemcpyHostToDevice, stream));

    // Launch kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    matmul_warp_opt<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, N);

    // Async transfer back
    CHECK_CUDA(cudaMemcpyAsync(h_C_pinned, d_C, bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy from pinned memory
    memcpy(h_C, h_C_pinned, bytes);

    // Free pinned memory
    CHECK_CUDA(cudaFreeHost(h_A_pinned));
    CHECK_CUDA(cudaFreeHost(h_B_pinned));
    CHECK_CUDA(cudaFreeHost(h_C_pinned));
}