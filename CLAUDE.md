# CUDA Exercise Documentation Format Guide

## Purpose
This guide establishes a consistent documentation format for all CUDA exercise modules to ensure clarity, maintainability, and effective learning progression.

---

## Directory Structure Rules

### Basic Structure (Modules 11-15)
```
XX.Module_Name/
├── README.md
├── CMakeLists.txt
└── *.cu, *.h files
```

### Advanced Structure (Module 16+)
```
XX.Module_Name/
├── README.md
├── CMakeLists.txt
├── src/
│   ├── kernels/
│   │   └── *.cu (kernel implementations, reusable across parts)
│   └── part_specific/
│       └── *.cu (module-specific code)
└── test/
    └── unit/
        ├── kernels/
        │   └── test_*.cu (kernel tests, reusable across parts)
        └── part_specific/
            └── test_*.cu (module-specific tests)
```

---

## README.md Format Template

### Header Structure
```markdown
# 🎯 Part XX: Module Title
**Goal**: One-sentence clear objective describing what learners will achieve.

## Project Structure
```
XX.Module_Name/
├── README.md          - This documentation
├── CMakeLists.txt     - Build configuration
├── src/               - Source implementations
│   ├── kernel.cu      - Core CUDA kernels
│   └── utils.h        - Helper functions
└── test/              - Test files
    └── test_kernel.cu - Unit tests
```

## Quick Navigation
- [XX.1 Topic One](#xx1-topic-one)
- [XX.2 Topic Two](#xx2-topic-two)
- [Build & Run](#build--run)
- [Summary](#summary)

---
```

### Section Format Rules

#### Rule 1: Introduction Sentences
Every section MUST start with 1-2 sentences explaining the concept's purpose and relevance.

```markdown
## **XX.1 Section Title**

This section demonstrates [concept] which is essential for [purpose]. Understanding this enables [benefit/application].

### **XX.1.1 Subsection**

Brief explanation of what this specific aspect covers and why it matters.
```

#### Rule 2: Code Samples with File References
All code examples should reference actual source files and include brief explanations.

```markdown
### **XX.1.2 Implementation Example**

The following implementation demonstrates memory coalescing patterns. This code is available in `src/kernels/memory_patterns.cu`.

```cpp
// memory_patterns.cu - Demonstrates coalesced vs strided access
__global__ void coalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;  // Coalesced: adjacent threads access adjacent memory
    }
}
```

**Key Points:**
- Adjacent threads access consecutive memory locations
- Results in efficient memory transactions
- Source: `src/kernels/memory_patterns.cu:8-14`
```

#### Rule 3: Practical Examples
Each major concept should include a working example with expected output.

```markdown
### **XX.1.3 Practical Usage**

Here's how to use this pattern in practice. Full example in `src/examples/pattern_demo.cu`.

```cpp
// pattern_demo.cu - Complete working example
int main() {
    const int N = 1024;
    float *d_data;

    cuda_malloc(&d_data, N);
    coalesced_access<<<(N+255)/256, 256>>>(d_data, N);
    CHECK_KERNEL_LAUNCH();

    // Verify results...
    cuda_free(d_data);
    return 0;
}
```

**Expected Output:**
```
Memory bandwidth: 450 GB/s (95% efficiency)
Execution time: 0.05 ms
```
```

---

## Parent Directory README Rules

### Navigation Links
Parent directories (10.cuda_basic, 20.cuda_intermediate, 30.CUDA_Libraries) MUST maintain navigation links to all subdirectories.

```markdown
# CUDA Basic Modules

## Module Navigation

### Foundations
- [11. Foundations](11.Foundations/README.md) - CUDA architecture and concepts
- [12. Your First CUDA Kernel](12.Your_First_CUDA_Kernel/README.md) - Writing and launching kernels

### Development & Testing
- [13. Debugging CUDA in VSCode](13.Debugging_CUDA_in_VSCode/README.md) - Debug tools and techniques
- [14. Code Inspection and Profiling](14.Code_Inspection_and_Profiling/README.md) - Performance analysis
- [15. Unit Testing](15.Unit_Testing/README.md) - Testing CUDA code

### Advanced Topics
- [16. Error Handling and Debugging](16.Error_Handling_and_Debugging/README.md) - Robust error management
- [17. Memory Hierarchy](17.Memory_Hierarchy/README.md) - Memory optimization
```

---

## Advanced Implementation Requirements

### Matrix Multiplication Example (Module 17+)
All advanced modules should include optimized implementations with progression from naive to optimized.

```markdown
## **17.3 Matrix Multiplication Evolution**

This section demonstrates the progression from naive to highly optimized matrix multiplication. Each implementation builds upon previous optimizations.

### **17.3.1 Naive Implementation**

Basic implementation without optimizations. Source: `src/kernels/matmul_naive.cu`.

```cpp
// matmul_naive.cu - O(N³) baseline implementation
__global__ void matmul_naive(float* C, const float* A, const float* B, int N) {
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
```

**Performance**: ~50 GFLOPS on RTX 3090

### **17.3.2 Tiled Implementation**

Uses shared memory for data reuse. Source: `src/kernels/matmul_tiled.cu`.

```cpp
// matmul_tiled.cu - Tiled implementation with shared memory
template<int TILE_SIZE>
__global__ void matmul_tiled(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Implementation details in source file
    // Key optimization: Reduces global memory accesses by TILE_SIZE factor
}
```

**Performance**: ~400 GFLOPS on RTX 3090 (8x improvement)

### **17.3.3 Tensor Core Implementation**

Leverages specialized hardware. Source: `src/kernels/matmul_wmma.cu`.

```cpp
// matmul_wmma.cu - Tensor Core accelerated implementation
#include <mma.h>
using namespace nvcuda;

__global__ void matmul_wmma(half* C, const half* A, const half* B, int N) {
    // Tensor Core implementation
    // Achieves near-peak throughput
}
```

**Performance**: ~10 TFLOPS on RTX 3090 (200x improvement over naive)
```

### Backpropagation Example (For Neural Network Modules)
```markdown
## **25.4 Backpropagation Implementation**

Implementing efficient backpropagation for neural network training. This demonstrates gradient computation and weight updates.

### **25.4.1 Forward Pass**

Source: `src/kernels/forward_pass.cu`

```cpp
// forward_pass.cu - Neural network forward propagation
__global__ void linear_forward(
    float* output,      // [batch_size, out_features]
    const float* input, // [batch_size, in_features]
    const float* weight,// [out_features, in_features]
    const float* bias,  // [out_features]
    int batch_size, int in_features, int out_features
) {
    // Compute y = xW^T + b
    // Full implementation in source file
}
```

### **25.4.2 Backward Pass**

Source: `src/kernels/backward_pass.cu`

```cpp
// backward_pass.cu - Gradient computation and backpropagation
__global__ void linear_backward(
    float* grad_input,  // [batch_size, in_features]
    float* grad_weight, // [out_features, in_features]
    float* grad_bias,   // [out_features]
    const float* grad_output, // [batch_size, out_features]
    const float* input,       // [batch_size, in_features]
    const float* weight,      // [out_features, in_features]
    int batch_size, int in_features, int out_features
) {
    // Compute gradients:
    // grad_input = grad_output @ weight
    // grad_weight = grad_output^T @ input
    // grad_bias = sum(grad_output, dim=0)
}
```

### **25.4.3 PyTorch Comparison**

Benchmark against PyTorch for validation. Test: `test/integration/test_torch_parity.cu`

```cpp
// test_torch_parity.cu - Validate against PyTorch
TEST(Backprop, ParityWithPyTorch) {
    // Compare our CUDA implementation with torch.nn.Linear
    // Ensures numerical accuracy within tolerance
}
```

**Performance Comparison:**
- Our CUDA: 850 GB/s memory bandwidth
- PyTorch: 820 GB/s memory bandwidth
- Advantage: Direct control over memory access patterns
```

---

## Testing Requirements

### Unit Test Format
All modules from 16 onwards must include comprehensive tests.

```markdown
## **XX.6 Testing**

### **XX.6.1 Unit Tests**

Tests are in `test/unit/`. Run with `ctest` or directly.

```cpp
// test/unit/test_kernel.cu
#include <gtest/gtest.h>
#include "gpu_gtest.h"

GPU_TEST(KernelTest, CorrectnessCheck) {
    // Test implementation
    const int N = 1024;
    float* d_data = cuda_malloc<float>(N);

    kernel<<<(N+255)/256, 256>>>(d_data, N);
    CHECK_KERNEL_LAUNCH();

    // Verify results
    GPU_EXPECT_NEAR(d_data[0], expected_value, 1e-5f);

    cuda_free(d_data);
}
```

### **XX.6.2 Performance Tests**

```cpp
// test/integration/test_performance.cu
TEST(Performance, BandwidthTest) {
    CudaTimer timer;
    const size_t size = 1 << 20;  // 1M elements

    timer.start();
    kernel<<<grid, block>>>(d_data, size);
    timer.stop();

    float bandwidth = calculate_bandwidth_gb(size * sizeof(float), timer.elapsed_ms());
    EXPECT_GT(bandwidth, 400.0f);  // Expect > 400 GB/s
}
```
```

---

## Build System Requirements

### CMakeLists.txt Template
```cmake
project(XX_Module_Name)

# Source files
add_library(${PROJECT_NAME}_lib
    src/kernels/kernel1.cu
    src/kernels/kernel2.cu
    src/utils/helpers.cu
)

# Example executables
add_executable(${PROJECT_NAME}_demo
    src/examples/demo.cu
)
target_link_libraries(${PROJECT_NAME}_demo ${PROJECT_NAME}_lib)

# Tests
if(BUILD_TESTING)
    add_executable(${PROJECT_NAME}_test
        test/unit/test_kernel1.cu
        test/unit/test_kernel2.cu
    )
    target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}_lib
        GTest::gtest_main
        GTestCudaGenerator
    )
    gtest_discover_tests(${PROJECT_NAME}_test)
endif()

# Profiling targets
add_profiling_targets(${PROJECT_NAME}_demo)
```

---

## Summary Section Template

Every README must end with a summary section:

```markdown
## **XX.7 Summary**

### **Key Takeaways**
1. Main concept learned and its importance
2. Performance optimization technique mastered
3. Common pitfall avoided through proper implementation

### **Performance Metrics**
- Baseline: X GFLOPS
- Optimized: Y GFLOPS
- Efficiency: Z% of theoretical peak

### **Common Errors & Solutions**
| Error | Cause | Solution |
|-------|-------|----------|
| Out of bounds | Missing boundary check | Add `if (idx < N)` |
| Race condition | Missing synchronization | Add `__syncthreads()` |

### **Next Steps**
- 📚 Continue to [Part XX+1: Next Topic](../XX.Next_Module/README.md)
- 🔧 Try exercises in `src/exercises/`
- 📊 Run performance benchmarks in `test/performance/`

### **References**
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
```

---

## Special Sections for Advanced Topics

### GPU-NVMe Interaction 
Must include:
1. NVMe user space IO queue and read data.
2. Set pinned memory on GPU and implement make IO queue on GPU.
3. Read data from NVMe to pinned memory on GPU.
4. make api take dict(key is kind, each kind start lba and length) and kind, and idx and length and pinned memory pointer, and read data from NVMe to pinned memory on GPU.
5. pycuda wrapper for above api.
6. C API Interface for above api.
7. Pytorch native cuda interface for above api.


### GPU Optimization
1. CPU implementation mat multiplication with pycuda
 > Native cpu implementation for mat multiplication for comparison.
2. CPU implementation backpropagation with pycuda
 > Native cpu implementation for backpropagation for comparison.
3. CPU implementation for attention layers with pycuda
 > Native cpu implementation for attention layers for comparison.
4. CPU implementation of experts with pycuda.
 > Native cpu implementation of experts for comparison.
5. load data from NVMe to CPU memory with pycuda.
6. C API Interface migration for pytorch from pycuda.
7. Pytorch native cuda interface migration.
8. Progressive GPU optimizations
9. Memory usage analysis


---

## Documentation Quality Checklist

- [ ] Every section has 1-2 sentence introduction
- [ ] All code examples reference actual source files
- [ ] Parent README has navigation links to all children
- [ ] Module 16+ has src/ and test/ directories
- [ ] Tests include both correctness and performance
- [ ] Summary section with metrics and next steps
- [ ] CMakeLists.txt follows standard template
- [ ] Performance comparisons with theoretical limits
- [ ] Common errors and solutions documented
- [ ] Build and run instructions are complete

---

## Example Excellence: Matrix Multiplication

The matrix multiplication implementation should serve as the gold standard, showing:
1. **Progression**: Naive → Tiled → Vectorized → Tensor Core
2. **Metrics**: GFLOPS at each optimization level
3. **Analysis**: Roofline model showing bandwidth vs compute limits
4. **Testing**: Validation against cuBLAS
5. **Documentation**: Clear explanation of each optimization

This demonstrates mastery of:
- Memory hierarchy (global → shared → registers)
- Parallelization strategies
- Hardware utilization (Tensor Cores)
- Performance analysis
- Production-quality testing


## Additional Notes
- Use ninja over make.
- Use cuda  13.0 or above.
- Use C++20 standard.
- Use GoogleTest for testing.
- Use Parameterized tests where applicable. and use 00.test_lib/gpu_gtest.h|gtest_generator.h for GPU tests. and parameterized tests. 
- Use 00.cuda_custom_lib/cuda_utils.h for cuda error check and cuda memory allocation.
- Use CUDA_BASIC_LIB and GTEST_BASIC_LIB for cuda and test related builds. rather than specifying cuda and gtest related libraries in each module.
- each section should have a couple of sentences of explanation about it.(whenever # or ## or ### is used)