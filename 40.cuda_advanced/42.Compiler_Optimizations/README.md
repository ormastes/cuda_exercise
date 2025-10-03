# 44. Compiler Optimizations and Code Generation

## 44.1 Introduction to CUDA Compiler Optimizations

CUDA compiler optimizations involve understanding how nvcc generates efficient GPU code and using compiler directives, flags, and techniques to maximize performance while managing resource constraints.

## 44.2 Learning Objectives

- Master CUDA compiler flags and optimization levels
- Understand register allocation and pressure management
- Implement efficient kernel launch bounds
- Optimize instruction-level parallelism (ILP)
- Analyze compiler-generated PTX and SASS code
- Use compiler intrinsics for hardware-specific optimizations

## 44.3 Compiler Fundamentals

### 44.3.1 NVCC Compilation Pipeline

```cpp
class CompilerAnalyzer {
private:
    std::string cudaArch;
    std::vector<std::string> compilerFlags;

public:
    CompilerAnalyzer(const std::string& arch = "sm_86") : cudaArch(arch) {
        setupOptimizationFlags();
    }

    void setupOptimizationFlags() {
        // Basic optimization flags
        compilerFlags = {
            "-O3",                          // Maximum optimization
            "-use_fast_math",               // Fast math operations
            "-Xptxas -v",                   // Verbose PTX assembler
            "--expt-relaxed-constexpr",     // Relaxed constexpr rules
            "-lineinfo",                    // Line info for profiling
            "--generate-line-info",         // Debug line information
            "-Xcompiler -ffast-math",       // Host compiler fast math
            "-maxrregcount=32"              // Limit registers per thread
        };
    }

    void analyzeCompilation(const std::string& sourceFile) {
        printf("Analyzing compilation for: %s\n", sourceFile.c_str());
        printf("Target architecture: %s\n", cudaArch.c_str());

        // Generate PTX for analysis
        std::string ptxCommand = "nvcc -ptx -arch=" + cudaArch + " " + sourceFile;
        printf("PTX generation: %s\n", ptxCommand.c_str());

        // Generate SASS for analysis
        std::string sassCommand = "nvcc -cubin -arch=" + cudaArch + " " + sourceFile;
        printf("SASS generation: %s\n", sassCommand.c_str());

        // Compile with optimization flags
        std::string optimizedCommand = "nvcc -O3 -use_fast_math -arch=" + cudaArch + " " + sourceFile;
        printf("Optimized compilation: %s\n", optimizedCommand.c_str());
    }

    void showRegisterUsage(const std::string& kernelName) {
        printf("\nRegister Usage Analysis for %s:\n", kernelName.c_str());
        printf("  Use: ptxas -v -arch=%s kernel.ptx\n", cudaArch.c_str());
        printf("  Look for: 'Used X registers'\n");
        printf("  Monitor: Register spilling warnings\n");
    }
};
```

### 44.3.2 Launch Bounds Optimization

```cuda
// Optimal launch bounds example
template<int BLOCK_SIZE, int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
void optimized_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compiler knows exact block size and can optimize accordingly
    __shared__ float sdata[BLOCK_SIZE];

    if (idx < n) {
        sdata[threadIdx.x] = data[idx];
    }
    __syncthreads();

    // Process with known block dimensions
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

// Register pressure management
class RegisterOptimizer {
public:
    // Method 1: Use launch bounds to control register allocation
    __global__ __launch_bounds__(256, 2)  // 256 threads, min 2 blocks per SM
    void register_limited_kernel(float* data) {
        // Compiler will limit register usage to fit 2 blocks per SM
        float local_data[4];  // Careful with local arrays

        // Use register reuse patterns
        float temp = load_data(data);
        temp = process_step1(temp);  // Reuse same register
        temp = process_step2(temp);
        store_data(data, temp);
    }

    // Method 2: Manual register management
    __global__ void manual_register_management(float* data) {
        // Group related variables to encourage register reuse
        struct {
            float x, y, z, w;
        } vector_data;

        // Use bit packing for flags
        unsigned int flags = 0;

        // Minimize register lifetime
        {
            float temp = expensive_computation();
            process_immediate(temp);
            // temp register freed here
        }

        // Use shared memory for large local arrays
        __shared__ float shared_buffer[256];
        shared_buffer[threadIdx.x] = data[threadIdx.x];
    }
};
```

## 44.4 Advanced Optimization Techniques

### 44.4.1 Instruction-Level Parallelism (ILP)

```cuda
template<int UNROLL_FACTOR>
class ILPOptimizer {
public:
    // Unrolled computation for ILP
    __device__ __forceinline__
    void vectorized_operation(float* input, float* output, int idx) {
        // Manual unrolling for instruction-level parallelism
        float4 data1 = reinterpret_cast<float4*>(input)[idx];
        float4 data2 = reinterpret_cast<float4*>(input)[idx + 1];

        // Process multiple values simultaneously
        float4 result1, result2;
        result1.x = sqrtf(data1.x * data1.x + 1.0f);
        result1.y = sqrtf(data1.y * data1.y + 1.0f);
        result1.z = sqrtf(data1.z * data1.z + 1.0f);
        result1.w = sqrtf(data1.w * data1.w + 1.0f);

        result2.x = sqrtf(data2.x * data2.x + 1.0f);
        result2.y = sqrtf(data2.y * data2.y + 1.0f);
        result2.z = sqrtf(data2.z * data2.z + 1.0f);
        result2.w = sqrtf(data2.w * data2.w + 1.0f);

        reinterpret_cast<float4*>(output)[idx] = result1;
        reinterpret_cast<float4*>(output)[idx + 1] = result2;
    }

    // Template-based unrolling
    template<int N>
    __device__ __forceinline__
    void unrolled_loop(float* data, int base_idx) {
        #pragma unroll N
        for (int i = 0; i < N; i++) {
            data[base_idx + i] = expensive_function(data[base_idx + i]);
        }
    }
};

// Kernel with maximum ILP
__global__ void ilp_optimized_kernel(float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;  // Process 8 elements per thread

    if (idx + 7 < n) {
        // Load multiple values for ILP
        float a0 = input[idx + 0], a1 = input[idx + 1];
        float a2 = input[idx + 2], a3 = input[idx + 3];
        float a4 = input[idx + 4], a5 = input[idx + 5];
        float a6 = input[idx + 6], a7 = input[idx + 7];

        // Independent computations for ILP
        float r0 = a0 * a0 + 1.0f;
        float r1 = a1 * a1 + 1.0f;
        float r2 = a2 * a2 + 1.0f;
        float r3 = a3 * a3 + 1.0f;
        float r4 = a4 * a4 + 1.0f;
        float r5 = a5 * a5 + 1.0f;
        float r6 = a6 * a6 + 1.0f;
        float r7 = a7 * a7 + 1.0f;

        // Store results
        output[idx + 0] = r0; output[idx + 1] = r1;
        output[idx + 2] = r2; output[idx + 3] = r3;
        output[idx + 4] = r4; output[idx + 5] = r5;
        output[idx + 6] = r6; output[idx + 7] = r7;
    }
}
```

### 44.4.2 Compiler Intrinsics and Fast Math

```cuda
class IntrinsicsOptimizer {
public:
    // Fast math intrinsics
    __device__ __forceinline__
    float fast_operations(float x, float y) {
        // Use fast intrinsics when precision allows
        float result = __fmaf_rn(x, y, 1.0f);  // Fused multiply-add
        result = __expf(result);                // Fast exponential
        result = __logf(result);                // Fast logarithm
        result = rsqrtf(result);                // Fast reciprocal sqrt
        return result;
    }

    // Bit manipulation intrinsics
    __device__ __forceinline__
    unsigned int bit_operations(unsigned int x) {
        unsigned int result = __brev(x);        // Bit reverse
        result = __popc(result);                // Population count
        result = __clz(result);                 // Count leading zeros
        result = __ffs(result);                 // Find first set bit
        return result;
    }

    // SIMD-style operations
    __device__ __forceinline__
    void simd_operations(float4& a, float4& b) {
        // Compiler can vectorize these operations
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }

    // Texture memory intrinsics
    __device__ __forceinline__
    float texture_fetch_optimized(cudaTextureObject_t tex, float x, float y) {
        // Hardware-accelerated texture interpolation
        return tex2D<float>(tex, x, y);
    }
};

// Demonstration kernel using intrinsics
__global__ void intrinsics_demo_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];

        // Use compiler intrinsics for optimal code generation
        float result = __fmaf_rn(x, 2.0f, 1.0f);  // x * 2 + 1
        result = __expf(result);                   // e^result
        result = __logf(result + 1.0f);           // log(result + 1)

        output[idx] = result;
    }
}
```

## 44.5 Practical Examples

### 44.5.1 Matrix Multiplication with Compiler Optimizations

```cuda
template<int TILE_SIZE>
class OptimizedMatMul {
private:
    static constexpr int BLOCK_SIZE = TILE_SIZE;

public:
    // Highly optimized matrix multiplication
    __global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 2)
    void optimized_matmul(const float* A, const float* B, float* C,
                         int M, int N, int K) {
        // Shared memory with optimal bank access patterns
        __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
        __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        float sum = 0.0f;

        // Tiled computation with register blocking
        for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
            // Collaborative loading with bounds checking
            int tileRow = tile * TILE_SIZE + threadIdx.x;
            int tileCol = tile * TILE_SIZE + threadIdx.y;

            tileA[threadIdx.y][threadIdx.x] =
                (row < M && tileRow < K) ? A[row * K + tileRow] : 0.0f;
            tileB[threadIdx.y][threadIdx.x] =
                (tileCol < K && col < N) ? B[tileCol * N + col] : 0.0f;

            __syncthreads();

            // Compute with unrolled inner loop for ILP
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum = __fmaf_rn(tileA[threadIdx.y][k], tileB[k][threadIdx.x], sum);
            }

            __syncthreads();
        }

        // Store result with bounds checking
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
};
```

### 44.5.2 Compiler-Optimized Reduction

```cuda
template<int BLOCK_SIZE>
class OptimizedReduction {
public:
    __global__ __launch_bounds__(BLOCK_SIZE, 4)
    void compiler_optimized_reduction(const float* input, float* output, int n) {
        __shared__ float sdata[BLOCK_SIZE];

        int tid = threadIdx.x;
        int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

        // Initial load with bounds checking and first reduction step
        float sum = 0.0f;
        if (idx < n) sum += input[idx];
        if (idx + BLOCK_SIZE < n) sum += input[idx + BLOCK_SIZE];

        sdata[tid] = sum;
        __syncthreads();

        // Unrolled reduction for known block sizes
        if constexpr (BLOCK_SIZE >= 512) {
            if (tid < 256) sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128) sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64) sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }

        // Final warp reduction using shuffle
        if (tid < 32) {
            float val = sdata[tid];
            val += sdata[tid + 32];
            val += __shfl_down_sync(0xffffffff, val, 16);
            val += __shfl_down_sync(0xffffffff, val, 8);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 1);

            if (tid == 0) output[blockIdx.x] = val;
        }
    }
};
```

## 44.6 Exercises

1. **Compiler Flag Analysis**: Compare performance with different optimization flags
2. **Register Pressure Study**: Analyze impact of register usage on occupancy
3. **ILP Optimization**: Implement and measure instruction-level parallelism
4. **Launch Bounds Tuning**: Find optimal launch bounds for different kernels
5. **Intrinsics Performance**: Compare compiler intrinsics vs standard functions

## 44.7 Building and Running

```bash
# Build with various optimization levels
cd build/30.cuda_advanced/44.Compiler_Optimizations
ninja

# Compile with different optimization flags
nvcc -O0 -arch=sm_86 basic_kernel.cu -o kernel_O0
nvcc -O3 -use_fast_math -arch=sm_86 basic_kernel.cu -o kernel_O3

# Analyze PTX output
nvcc -ptx -O3 -arch=sm_86 kernel.cu
ptxas -v -arch=sm_86 kernel.ptx

# Analyze SASS output
nvcc -cubin -O3 -arch=sm_86 kernel.cu
cuobjdump -sass kernel.cubin

# Profile register usage
ncu --metrics local_memory_overhead,achieved_occupancy ./44_CompilerOptimizations

# Compare optimization levels
nsys profile --stats=true -t cuda ./kernel_O0
nsys profile --stats=true -t cuda ./kernel_O3
```

## 44.8 Key Takeaways

- Compiler optimization flags significantly impact performance
- Launch bounds help control register allocation and occupancy
- Instruction-level parallelism improves computational throughput
- Compiler intrinsics provide access to hardware-specific optimizations
- Register pressure management is crucial for high occupancy
- PTX and SASS analysis guides low-level optimization decisions
