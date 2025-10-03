# 34. Tensor Cores

## 34.1 Introduction

Tensor Cores provide dedicated hardware for mixed-precision matrix operations, accelerating AI and HPC workloads.

## 34.2 Learning Objectives

- Understand Tensor Core architecture
- Use WMMA API for matrix operations
- Implement mixed-precision algorithms
- Optimize for Tensor Core utilization

## 34.3 Tensor Core Basics

### 34.3.1 Hardware Overview

Tensor Cores perform:
- Matrix multiply-accumulate: D = A×B + C
- Mixed precision: FP16 inputs, FP32 accumulate
- Available on SM 7.0+ (Volta and newer)
- 8× throughput vs CUDA cores

### 34.3.2 Supported Operations

```cuda
// Supported matrix sizes (M×N×K)
// 16×16×16 (Volta, Turing)
// 8×8×4 (Ampere - sub-matrix)
// Various types: FP16, BF16, TF32, INT8, INT4
```

## 34.4 WMMA API

### 34.4.1 Basic WMMA Example

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_example(half* a, half* b, float* c) {
    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrices
    int lda = 16;
    int ldb = 16;
    wmma::load_matrix_sync(a_frag, a, lda);
    wmma::load_matrix_sync(b_frag, b, ldb);

    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    int ldc = 16;
    wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_row_major);
}
```

### 34.4.2 Tiled Matrix Multiplication

```cuda
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void tiled_wmma_gemm(
    half* a, half* b, float* c,
    int M, int N, int K) {

    // Warp and lane identification
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load tiles
            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // Perform multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(c + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}
```

## 34.5 Advanced Tensor Core Features

### 34.5.1 Mixed Precision Types

```cuda
// TF32 for single precision
__global__ void tf32_gemm(float* a, float* b, float* c) {
    // Automatic TF32 usage in compute capability 8.0+
    // 10-bit mantissa, 8-bit exponent

    // Enable/disable TF32
    // cudaSetMathMode(CUDA_TF32_TENSOR_OP_MATH);
}

// BF16 support
__global__ void bf16_wmma(
    __nv_bfloat16* a,
    __nv_bfloat16* b,
    float* c) {

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Load and compute with BF16
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```

### 34.5.2 Integer Operations

```cuda
__global__ void int8_wmma(
    signed char* a,
    signed char* b,
    int* c) {

    // INT8 WMMA operations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```

## 34.6 Optimization Strategies

### 34.6.1 Data Layout Optimization

```cuda
// Optimize memory layout for Tensor Cores
template<typename T>
__global__ void optimize_layout(T* input, T* output, int rows, int cols) {
    // Pad to multiple of 16 for optimal Tensor Core usage
    int padded_rows = (rows + 15) / 16 * 16;
    int padded_cols = (cols + 15) / 16 * 16;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows * cols) {
        int row = tid / cols;
        int col = tid % cols;

        // Rearrange for coalesced access
        output[row * padded_cols + col] = input[tid];
    }
}
```

### 34.6.2 Pipeline Optimization

```cuda
__global__ void pipelined_wmma(half* a, half* b, float* c, int K) {
    // Double buffering for fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Prefetch first tiles
    wmma::load_matrix_sync(a_frag[0], a, 16);
    wmma::load_matrix_sync(b_frag[0], b, 16);

    for (int k = 0; k < K - 16; k += 16) {
        // Load next tiles while computing current
        wmma::load_matrix_sync(a_frag[(k/16 + 1) % 2], a + k + 16, 16);
        wmma::load_matrix_sync(b_frag[(k/16 + 1) % 2], b + k + 16, 16);

        // Compute with current tiles
        wmma::mma_sync(acc_frag, a_frag[k/16 % 2], b_frag[k/16 % 2], acc_frag);
    }

    // Last computation
    wmma::mma_sync(acc_frag, a_frag[(K-16)/16 % 2], b_frag[(K-16)/16 % 2], acc_frag);

    wmma::store_matrix_sync(c, acc_frag, 16, wmma::mem_row_major);
}
```

## 34.7 Practical Applications

### 34.7.1 Deep Learning Inference

```cuda
class TensorCoreConvolution {
    // Optimized convolution using Tensor Cores
public:
    __device__ void im2col_tc(
        half* input, half* kernel, float* output,
        int H, int W, int C, int K, int R, int S) {

        // Transform input to column matrix
        // Apply WMMA for matrix multiplication
        // Transform back to image format
    }
};
```

### 34.7.2 Mixed Precision Training

```cuda
__global__ void mixed_precision_sgd(
    half* weights,
    half* gradients,
    float* master_weights,
    float learning_rate,
    int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Update master weights in FP32
        float weight = __half2float(weights[tid]);
        float grad = __half2float(gradients[tid]);

        master_weights[tid] -= learning_rate * grad;

        // Convert back to FP16 for computation
        weights[tid] = __float2half(master_weights[tid]);
    }
}
```

## 34.8 Performance Profiling

### 34.8.1 Tensor Core Utilization

```cuda
// Check Tensor Core usage with Nsight Compute
// ncu --metrics sm__pipe_tensor_* ./tensor_core_app

void measure_tensor_core_performance() {
    // Regular CUDA cores GEMM
    float cuda_time = measure_cuda_gemm();

    // Tensor Core GEMM
    float tc_time = measure_tc_gemm();

    printf("Speedup: %.2fx\n", cuda_time / tc_time);
}
```

## 34.9 Exercises

1. **Basic WMMA**: Implement matrix multiplication using WMMA API
2. **Mixed Precision**: Compare FP16, BF16, and TF32 performance
3. **Optimization**: Optimize data layout for Tensor Cores
4. **Application**: Implement convolution using Tensor Cores

## 34.10 Building and Running

```bash
# Build Tensor Core examples
cd build/30.cuda_advanced/34.Tensor_Cores
ninja

# Run examples (requires GPU with Tensor Cores)
./34_TensorCores_wmma_basic
./34_TensorCores_mixed_precision
./34_TensorCores_optimized

# Profile Tensor Core utilization
ncu --set full --metrics sm__pipe_tensor_cycles_active ./34_TensorCores_wmma_basic
```

## 34.11 Key Takeaways

- Tensor Cores provide massive speedup for matrix operations
- WMMA API enables direct Tensor Core programming
- Mixed precision balances performance and accuracy
- Data layout critical for performance
- Essential for modern AI workloads