# 46. Tile-Based Programming Model (CUDA 13.0+)

## 46.1 Introduction

The Tile-Based Programming Model, introduced in CUDA 13.0, provides a complementary approach to traditional thread-based programming, allowing developers to express operations on entire arrays or matrices while the system handles low-level execution details.

## 46.2 Learning Objectives

- Understand tile-based vs thread-based programming models
- Implement tile-based algorithms
- Use CUDA's tile API effectively
- Optimize memory access with tiles
- Migrate from thread-based to tile-based code

## 46.3 Tile-Based Programming Fundamentals

### 46.3.1 What is Tile-Based Programming?

Tile-based programming allows you to:
- Think in terms of data tiles rather than individual threads
- Express operations on entire arrays/matrices
- Let the system handle thread-level details
- Similar to NumPy/array programming in Python
- Focus on algorithms rather than thread management

### 46.3.2 Comparison with Thread-Based Model

```cuda
// Traditional Thread-Based Approach
__global__ void thread_based_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Tile-Based Approach (CUDA 13.0+)
namespace cuda::tile {

template<typename TileShape>
__global__ void tile_based_add(TiledArray<float, TileShape> a,
                               TiledArray<float, TileShape> b,
                               TiledArray<float, TileShape> c) {
    // Get the tile this thread block is responsible for
    auto tile = get_tile<TileShape>();

    // Load tiles into shared memory
    auto tile_a = tile.load(a);
    auto tile_b = tile.load(b);

    // Perform tile-wise operation
    auto tile_c = tile_a + tile_b;

    // Store result tile
    tile.store(c, tile_c);
}

} // namespace cuda::tile
```

## 46.4 Tile API and Operations

### 46.4.1 Basic Tile Operations

```cuda
#include <cuda/tile.h>

using namespace cuda::tile;

// Define tile shape
using TileShape = Shape<16, 16>;  // 16x16 tile

template<typename T>
class TileOperations {
public:
    // Tile-based matrix multiplication
    __device__ static void tile_gemm(
        Tile<T, TileShape>& tile_c,
        const Tile<T, TileShape>& tile_a,
        const Tile<T, TileShape>& tile_b) {

        // Tile-level matrix multiply
        tile_c = tile_a.matmul(tile_b);
    }

    // Tile-based reduction
    __device__ static T tile_reduce(const Tile<T, TileShape>& tile) {
        return tile.reduce_sum();
    }

    // Tile-based transpose
    __device__ static void tile_transpose(
        Tile<T, TileShape>& output,
        const Tile<T, TileShape>& input) {

        output = input.transpose();
    }
};
```

### 46.4.2 Advanced Tile Patterns

```cuda
// Hierarchical Tile Processing
template<int TILE_M, int TILE_N, int TILE_K>
class HierarchicalTiles {
    using OuterTile = Shape<TILE_M, TILE_N>;
    using InnerTile = Shape<8, 8>;  // Sub-tiles within outer tile

public:
    __device__ void process_hierarchical(
        TiledArray<float, OuterTile>& output,
        const TiledArray<float, OuterTile>& input) {

        // Get outer tile
        auto outer = get_tile<OuterTile>();

        // Partition into inner tiles
        auto inner_tiles = outer.partition<InnerTile>();

        // Process each inner tile
        for (auto& inner : inner_tiles) {
            auto data = inner.load(input);
            data = process_inner_tile(data);
            inner.store(output, data);
        }
    }

private:
    __device__ Tile<float, InnerTile> process_inner_tile(
        const Tile<float, InnerTile>& tile) {
        // Apply operation to inner tile
        return tile.apply([](float x) { return sqrtf(x * x + 1.0f); });
    }
};
```

## 46.5 Memory Management with Tiles

### 46.5.1 Automatic Memory Optimization

```cuda
template<typename T, int TILE_SIZE>
class TileMemoryManager {
public:
    // Tiles automatically handle shared memory allocation
    __device__ void optimized_stencil(
        TiledArray<T, Shape<TILE_SIZE, TILE_SIZE>>& output,
        const TiledArray<T, Shape<TILE_SIZE, TILE_SIZE>>& input) {

        // Automatic halo loading for stencil operations
        constexpr int HALO = 2;
        using StencilTile = Shape<TILE_SIZE + 2*HALO, TILE_SIZE + 2*HALO>;

        auto tile = get_tile<StencilTile>();

        // Load with halos - handled automatically
        auto stencil_data = tile.load_with_halo(input, HALO);

        // Apply stencil operation
        auto result = apply_stencil(stencil_data);

        // Store without halos
        tile.store_interior(output, result, HALO);
    }

private:
    __device__ Tile<T, Shape<TILE_SIZE, TILE_SIZE>>
    apply_stencil(const Tile<T, Shape<TILE_SIZE + 4, TILE_SIZE + 4>>& input) {
        Tile<T, Shape<TILE_SIZE, TILE_SIZE>> output;

        // 5-point stencil
        for (int i = 2; i < TILE_SIZE + 2; i++) {
            for (int j = 2; j < TILE_SIZE + 2; j++) {
                output(i-2, j-2) = 0.2f * (
                    input(i, j) +
                    input(i-1, j) + input(i+1, j) +
                    input(i, j-1) + input(i, j+1)
                );
            }
        }

        return output;
    }
};
```

### 46.5.2 Tile Synchronization

```cuda
class TileSynchronization {
public:
    // Tile-level barriers
    __device__ void synchronized_tiles(
        TiledArray<float, Shape<32, 32>>& data,
        int num_iterations) {

        auto tile = get_tile<Shape<32, 32>>();

        for (int iter = 0; iter < num_iterations; iter++) {
            // Load tile
            auto local_tile = tile.load(data);

            // Process tile
            local_tile = process_iteration(local_tile, iter);

            // Tile-level synchronization
            tile.sync();

            // Store tile
            tile.store(data, local_tile);

            // Global tile synchronization
            tile.global_sync();
        }
    }

private:
    __device__ Tile<float, Shape<32, 32>>
    process_iteration(const Tile<float, Shape<32, 32>>& input, int iter) {
        return input.apply([iter](float x) {
            return x * (1.0f + 0.1f * iter);
        });
    }
};
```

## 46.6 Practical Examples

### 46.6.1 Tile-Based Matrix Multiplication

```cuda
template<int M, int N, int K>
class TileBasedGEMM {
    using TileA = Shape<M, K>;
    using TileB = Shape<K, N>;
    using TileC = Shape<M, N>;

public:
    __global__ void gemm_kernel(
        TiledMatrix<float, M, K> A,
        TiledMatrix<float, K, N> B,
        TiledMatrix<float, M, N> C,
        float alpha, float beta) {

        // Get tile coordinates
        auto tile_coord = get_tile_coord();

        // Allocate accumulator tile
        Tile<float, TileC> acc = 0.0f;

        // Iterate over K dimension in tiles
        for (int k_tile = 0; k_tile < A.num_tiles_k(); k_tile++) {
            // Load A and B tiles
            auto tile_a = A.load_tile(tile_coord.row, k_tile);
            auto tile_b = B.load_tile(k_tile, tile_coord.col);

            // Tile-level matrix multiply-accumulate
            acc = tile_a.matmul(tile_b, acc);
        }

        // Scale and add with existing C
        auto tile_c = C.load_tile(tile_coord.row, tile_coord.col);
        acc = alpha * acc + beta * tile_c;

        // Store result
        C.store_tile(tile_coord.row, tile_coord.col, acc);
    }
};
```

### 46.6.2 Tile-Based Convolution

```cuda
template<int TILE_H, int TILE_W, int FILTER_SIZE>
class TileBasedConvolution {
    using InputTile = Shape<TILE_H + FILTER_SIZE - 1, TILE_W + FILTER_SIZE - 1>;
    using OutputTile = Shape<TILE_H, TILE_W>;

public:
    __global__ void conv2d_kernel(
        TiledArray2D<float> input,
        TiledArray2D<float> output,
        Tile<float, Shape<FILTER_SIZE, FILTER_SIZE>> filter) {

        // Get output tile position
        auto tile_pos = get_tile_position<OutputTile>();

        // Load input tile with padding for filter
        auto input_tile = load_padded_tile<InputTile>(input, tile_pos);

        // Perform tile-based convolution
        auto output_tile = tile_convolution(input_tile, filter);

        // Store output tile
        store_tile<OutputTile>(output, tile_pos, output_tile);
    }

private:
    __device__ Tile<float, OutputTile> tile_convolution(
        const Tile<float, InputTile>& input,
        const Tile<float, Shape<FILTER_SIZE, FILTER_SIZE>>& filter) {

        Tile<float, OutputTile> output;

        // Tile-aware convolution
        for (int oh = 0; oh < TILE_H; oh++) {
            for (int ow = 0; ow < TILE_W; ow++) {
                float sum = 0.0f;

                // Apply filter
                for (int fh = 0; fh < FILTER_SIZE; fh++) {
                    for (int fw = 0; fw < FILTER_SIZE; fw++) {
                        sum += input(oh + fh, ow + fw) * filter(fh, fw);
                    }
                }

                output(oh, ow) = sum;
            }
        }

        return output;
    }
};
```

## 46.7 Migration from Thread-Based to Tile-Based

### 46.7.1 Migration Strategy

```cuda
// Original Thread-Based Code
__global__ void thread_reduce(float* data, float* result, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Migrated Tile-Based Code
template<int TILE_SIZE>
__global__ void tile_reduce(TiledArray<float, Shape<TILE_SIZE>> data,
                           float* result) {
    // Get tile for this block
    auto tile = get_tile<Shape<TILE_SIZE>>();

    // Load tile data
    auto tile_data = tile.load(data);

    // Tile-level reduction - system handles the details
    float tile_sum = tile_data.reduce_sum();

    // Store result (only one thread writes)
    if (tile.is_leader()) {
        result[tile.get_tile_id()] = tile_sum;
    }
}
```

## 46.8 Performance Considerations

### 46.8.1 Tile Size Selection

```cuda
class TileSizeOptimizer {
public:
    // Automatic tile size selection based on problem size
    template<typename T>
    static constexpr auto select_tile_size(size_t matrix_m, size_t matrix_n) {
        // Consider shared memory constraints
        constexpr size_t SHARED_MEM_SIZE = 48 * 1024;  // 48KB
        constexpr size_t ELEM_SIZE = sizeof(T);

        // Calculate optimal tile size
        if (matrix_m >= 256 && matrix_n >= 256) {
            return Shape<32, 32>();  // Large matrices
        } else if (matrix_m >= 128 && matrix_n >= 128) {
            return Shape<16, 16>();  // Medium matrices
        } else {
            return Shape<8, 8>();    // Small matrices
        }
    }

    // Runtime tile size adaptation
    __device__ void adaptive_tiling(float* data, int n) {
        if (n >= 1024) {
            process_with_tile<64>(data, n);
        } else if (n >= 256) {
            process_with_tile<32>(data, n);
        } else {
            process_with_tile<16>(data, n);
        }
    }

private:
    template<int TILE_SIZE>
    __device__ void process_with_tile(float* data, int n) {
        auto tile = get_tile<Shape<TILE_SIZE>>();
        // Process with selected tile size
    }
};
```

## 46.9 Exercises

1. **Basic Tile Operations**: Implement vector addition using tile-based model
2. **Tile Matrix Multiply**: Create efficient tiled GEMM
3. **Tile Stencil**: Implement 2D stencil with automatic halo handling
4. **Migration Practice**: Convert thread-based kernels to tile-based
5. **Performance Comparison**: Benchmark tile-based vs thread-based

## 46.10 Building and Running

```bash
# Build with CUDA 13.0+ support
cd build/30.cuda_advanced/46.Tile_Based_Programming
ninja

# Run examples
./46_TileBased_basic
./46_TileBased_gemm
./46_TileBased_convolution

# Profile tile-based kernels
ncu --set full ./46_TileBased_gemm

# Compare with thread-based version
./compare_thread_vs_tile
```

## 46.11 Key Takeaways

- Tile-based programming simplifies GPU programming by abstracting thread management
- Similar to array programming in high-level languages like NumPy
- Automatic memory management and optimization
- Easier to express complex algorithms
- Complementary to thread-based model, not a replacement
- Particularly effective for regular, structured computations
- Reduces boilerplate code and potential for errors