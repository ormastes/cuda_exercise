# 31. PTX Assembly

## 31.1 Introduction to PTX

PTX (Parallel Thread Execution) is CUDA's virtual assembly language that provides low-level control over GPU operations.

## 31.2 Learning Objectives

- Understand PTX assembly basics
- Write inline PTX in CUDA kernels
- Optimize critical code sections with PTX
- Debug and analyze PTX generation

## 31.3 PTX Basics

### 31.3.1 What is PTX?

PTX is an intermediate representation:
- Virtual ISA for NVIDIA GPUs
- Compiled to device-specific SASS
- Provides forward compatibility
- Enables low-level optimizations

### 31.3.2 PTX Syntax

```ptx
.version 7.0
.target sm_70
.address_size 64

// Function declaration
.entry kernel_name(
    .param .u64 param1,
    .param .u32 param2
) {
    // PTX instructions
    .reg .u32 %r<10>;  // Registers
    .reg .f32 %f<5>;   // Float registers

    ld.param.u64 %rd1, [param1];
    mov.u32 %r1, %tid.x;
    // ...
}
```

## 31.4 Inline PTX in CUDA

### 31.4.1 Basic Inline Assembly

```cuda
__device__ int add_ptx(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;"
        : "=r"(result)    // Output
        : "r"(a), "r"(b)  // Inputs
    );
    return result;
}
```

### 31.4.2 Complex Operations

```cuda
__device__ float fma_ptx(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c)
    );
    return result;
}

__device__ int popc_ptx(int x) {
    int result;
    asm("popc.b32 %0, %1;"
        : "=r"(result)
        : "r"(x)
    );
    return result;
}
```

## 31.5 Advanced PTX Techniques

### 31.5.1 Memory Operations

```cuda
__device__ void prefetch_l1(const void* ptr) {
    asm("prefetch.global.L1 [%0];" :: "l"(ptr));
}

__device__ void cache_streaming(float* dst, const float* src) {
    float val;
    asm("ld.global.cs.f32 %0, [%1];"
        : "=f"(val)
        : "l"(src)
    );
    asm("st.global.cs.f32 [%0], %1;"
        :: "l"(dst), "f"(val)
    );
}
```

### 31.5.2 Atomic Operations

```cuda
__device__ int atomic_min_ptx(int* addr, int val) {
    int old;
    asm("atom.global.min.s32 %0, [%1], %2;"
        : "=r"(old)
        : "l"(addr), "r"(val)
    );
    return old;
}
```

## 31.6 Practical Examples

### 31.6.1 Custom Reduction

```cuda
__device__ int warp_reduce_ptx(int val) {
    // Use shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp;
        asm("shfl.down.b32 %0, %1, %2, 0x1f;"
            : "=r"(temp)
            : "r"(val), "r"(offset)
        );
        val += temp;
    }
    return val;
}
```

### 31.6.2 Bit Manipulation

```cuda
__device__ int find_leading_one(int x) {
    int result;
    asm("bfind.u32 %0, %1;"
        : "=r"(result)
        : "r"(x)
    );
    return result;
}

__device__ int reverse_bits(int x) {
    int result;
    asm("brev.b32 %0, %1;"
        : "=r"(result)
        : "r"(x)
    );
    return result;
}
```

## 31.7 PTX Generation and Analysis

### 31.7.1 Generating PTX

```bash
# Generate PTX from CUDA source
nvcc -ptx kernel.cu -o kernel.ptx

# Keep intermediate files
nvcc -keep kernel.cu

# Specific architecture
nvcc -arch=sm_80 -ptx kernel.cu
```

### 31.7.2 Analyzing PTX

```cuda
// Force PTX generation for analysis
__global__ void __launch_bounds__(256, 2)
analyzed_kernel(float* data) {
    // Kernel code
}

// Check generated PTX
// cuobjdump -ptx executable
```

## 31.8 Performance Considerations

### 31.8.1 When to Use PTX

- Critical inner loops
- Accessing special registers
- Custom memory access patterns
- Hardware-specific features

### 31.8.2 PTX vs CUDA C++

```cuda
// CUDA C++ version
__device__ float safe_divide(float a, float b) {
    return a / b;  // May include checks
}

// PTX version - direct hardware division
__device__ float fast_divide(float a, float b) {
    float result;
    asm("div.approx.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(a), "f"(b)
    );
    return result;
}
```

## 31.9 Exercises

1. **Basic PTX Operations**: Implement common operations using inline PTX
2. **Performance Comparison**: Compare PTX vs CUDA C++ for specific operations
3. **Custom Instructions**: Use PTX for hardware-specific features
4. **PTX Analysis**: Analyze generated PTX for optimization opportunities

## 31.10 Building and Running

```bash
# Build with PTX examples
cd build/30.cuda_advanced/31.PTX_Assembly
ninja

# Run examples
./31_PTXAssembly_inline_ptx
./31_PTXAssembly_performance

# Analyze PTX generation
cuobjdump -ptx ./31_PTXAssembly_inline_ptx
nvdisasm -c ./31_PTXAssembly_inline_ptx
```

## 31.11 Key Takeaways

- PTX provides low-level GPU control
- Inline PTX enables specific optimizations
- Use for critical performance sections
- Understanding PTX helps optimization
- Balance between performance and maintainability