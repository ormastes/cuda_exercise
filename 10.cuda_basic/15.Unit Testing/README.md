# ⚡ Part 5: Unit Testing CUDA Code

**Goal**: Learn how to write unit tests for both host and device CUDA code using Google Test.

---

## **5.1 Introduction to GPU Testing**

Testing CUDA code presents unique challenges:
- **Device code** runs on GPU and cannot directly use CPU testing frameworks
- **Kernel launches** are asynchronous and need proper synchronization
- **Memory management** between host and device requires careful handling
- **Error checking** must cover both launch and execution errors

Our solution uses `gpu_gtest.h` - a lightweight bridge between Google Test and CUDA that enables:
- Host-level testing of CUDA API calls
- Device-level testing of kernel logic
- Unified test reporting and assertions

---

## **5.2 Host-Level Unit Testing**

Host-level tests verify CUDA API usage, memory management, and kernel launches from the CPU side.

### **5.2.1 Basic Structure**

```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(CudaApiTest, DeviceDetection) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    EXPECT_EQ(error, cudaSuccess);
    EXPECT_GT(deviceCount, 0) << "No CUDA devices found";
}
```

### **5.2.2 Testing Memory Operations**

```cpp
TEST(CudaMemoryTest, AllocateAndFree) {
    const size_t size = 1024 * sizeof(float);
    float* d_data = nullptr;

    // Test allocation
    EXPECT_EQ(cudaMalloc(&d_data, size), cudaSuccess);
    EXPECT_NE(d_data, nullptr);

    // Test memset
    EXPECT_EQ(cudaMemset(d_data, 0, size), cudaSuccess);

    // Test free
    EXPECT_EQ(cudaFree(d_data), cudaSuccess);
}
```

### **5.2.3 Testing Kernel Launches**

```cpp
__global__ void simpleKernel(int* result) {
    *result = threadIdx.x + blockIdx.x * blockDim.x;
}

TEST(KernelLaunchTest, SimpleKernelExecution) {
    int* d_result;
    int h_result = -1;

    ASSERT_EQ(cudaMalloc(&d_result, sizeof(int)), cudaSuccess);

    simpleKernel<<<1, 32>>>(d_result);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_GE(h_result, 0);
    EXPECT_LT(h_result, 32);

    cudaFree(d_result);
}
```

---

## **5.3 Device-Level Unit Testing**

Device-level tests run assertions directly inside CUDA kernels using our `gpu_gtest.h` bridge.

### **5.3.1 GPU_TEST Macro**

The simplest form runs a test kernel with default <<<1,1>>> configuration:

```cpp
#include "../../00.lib/gpu_gtest.h"

GPU_TEST(DeviceMath, BasicArithmetic) {
    int a = 5;
    int b = 3;

    GPU_EXPECT_EQ(a + b, 8);
    GPU_EXPECT_EQ(a - b, 2);
    GPU_EXPECT_EQ(a * b, 15);
    GPU_EXPECT_TRUE(a > b);
}
```

### **5.3.2 GPU_TEST_CFG with Custom Launch Configuration**

For tests requiring multiple threads:

```cpp
GPU_TEST_CFG(ParallelTest, ThreadIndexing, 2, 256) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread verifies its own index
    GPU_EXPECT_TRUE(tid >= 0);
    GPU_EXPECT_TRUE(tid < 512);

    // Test thread cooperation
    __shared__ int shared_data[256];
    shared_data[threadIdx.x] = threadIdx.x;
    __syncthreads();

    if (threadIdx.x == 0) {
        GPU_EXPECT_EQ(shared_data[0], 0);
        GPU_EXPECT_EQ(shared_data[255], 255);
    }
}
```

### **5.3.3 GPU_TEST_F with Fixtures**

For complex tests requiring setup/teardown:

```cpp
class VectorAddFixture : public ::testing::Test {
public:
    struct DeviceView {
        float* a;
        float* b;
        float* c;
        int n;
    };

protected:
    void SetUp() override {
        n = 1024;
        size_t size = n * sizeof(float);

        // Allocate unified memory
        cudaMallocManaged(&d_view, sizeof(DeviceView));
        cudaMallocManaged(&d_view->a, size);
        cudaMallocManaged(&d_view->b, size);
        cudaMallocManaged(&d_view->c, size);
        d_view->n = n;

        // Initialize data
        for (int i = 0; i < n; i++) {
            d_view->a[i] = i;
            d_view->b[i] = i * 2;
        }
    }

    void TearDown() override {
        cudaFree(d_view->a);
        cudaFree(d_view->b);
        cudaFree(d_view->c);
        cudaFree(d_view);
    }

    const DeviceView* device_view() const { return d_view; }

    GpuLaunchCfg launch_cfg() const {
        return MakeLaunchCfg((n + 255) / 256, 256);
    }

private:
    DeviceView* d_view;
    int n;
};

GPU_TEST_F(VectorAddFixture, VectorAddition) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < _ctx->n) {
        _ctx->c[idx] = _ctx->a[idx] + _ctx->b[idx];

        // Verify result
        float expected = idx + idx * 2;
        GPU_EXPECT_NEAR(_ctx->c[idx], expected, 0.001f);
    }
}
```

---

## **5.4 Available Device Assertions**

Inside device test bodies, you can use:

| Macro | Description | Example |
|-------|-------------|---------|
| `GPU_EXPECT_TRUE(cond)` | Check condition is true | `GPU_EXPECT_TRUE(x > 0)` |
| `GPU_ASSERT_TRUE(cond)` | Assert (stops test on failure) | `GPU_ASSERT_TRUE(ptr != nullptr)` |
| `GPU_EXPECT_EQ(a, b)` | Check equality | `GPU_EXPECT_EQ(result, 42)` |
| `GPU_EXPECT_NEAR(a, b, tol)` | Check floating-point near-equality | `GPU_EXPECT_NEAR(pi, 3.14159, 0.001)` |

### **5.4.1 Grid-Stride Loop Helper**

For testing over large datasets:

```cpp
GPU_TEST_CFG(LargeDataTest, ProcessArray, 256, 256) {

    const int N = 1000000;

    GPU_FOR_ALL(i, N) {
        // This loop handles any N with any grid/block size
        int value = i * 2;
        GPU_EXPECT_TRUE(value == i + i);
    }
}
```

---

## **5.5 Best Practices**

### **5.5.1 Error Checking Pattern**

Always check both launch and execution errors:

```cpp
TEST(KernelTest, ProperErrorChecking) {
    kernel<<<grid, block>>>(...);

    // Check launch error
    cudaError_t launchErr = cudaGetLastError();
    ASSERT_EQ(launchErr, cudaSuccess)
        << "Launch failed: " << cudaGetErrorString(launchErr);

    // Check execution error
    cudaError_t execErr = cudaDeviceSynchronize();
    ASSERT_EQ(execErr, cudaSuccess)
        << "Execution failed: " << cudaGetErrorString(execErr);
}
```

### **5.5.2 Memory Management**

Use RAII or fixtures for automatic cleanup:

```cpp
class CudaMemory {
    void* ptr = nullptr;
    size_t size = 0;
public:
    CudaMemory(size_t s) : size(s) {
        cudaMalloc(&ptr, size);
    }
    ~CudaMemory() {
        if (ptr) cudaFree(ptr);
    }
    void* get() { return ptr; }
};

TEST(MemoryTest, RAIIPattern) {
    CudaMemory mem(1024);
    ASSERT_NE(mem.get(), nullptr);
    // Automatic cleanup on scope exit
}
```

### **5.5.3 Test Organization**

- Group related tests in test suites
- Use descriptive test names
- Test both success and failure cases
- Verify edge cases and boundary conditions

---

## **5.6 Running the Examples**

This tutorial includes complete examples:

1. **host_test.cu** - Host-level testing examples
2. **device_test.cu** - Device-level testing with GPU_TEST macros
3. **CMakeLists.txt** - Build configuration

### Build and Run:

```bash
cd build
cmake ..
make cuda_unit_test
./10.cuda_basic/15.unit_test/cuda_unit_test
```

### Expected Output:

```
[==========] Running 12 tests from 4 test suites.
[----------] Global test environment set-up.
[----------] 3 tests from HostLevel
[ RUN      ] HostLevel.DeviceDetection
[       OK ] HostLevel.DeviceDetection (0 ms)
[ RUN      ] HostLevel.MemoryOperations
[       OK ] HostLevel.MemoryOperations (1 ms)
[ RUN      ] HostLevel.KernelLaunch
[       OK ] HostLevel.KernelLaunch (2 ms)
[----------] 3 tests from HostLevel (3 ms total)

[----------] 4 tests from DeviceLevel
[ RUN      ] DeviceLevel.BasicMath
[       OK ] DeviceLevel.BasicMath (1 ms)
...
[==========] 12 tests from 4 test suites ran. (15 ms total)
[  PASSED  ] 12 tests.
```

---

## **Summary**

You've learned how to:
- ✅ Write host-level tests for CUDA API calls
- ✅ Create device-level tests that run on GPU
- ✅ Use fixtures for complex test scenarios
- ✅ Handle CUDA-specific testing challenges
- ✅ Organize and run comprehensive CUDA test suites

Unit testing ensures your CUDA code is reliable, maintainable, and performs as expected across different configurations and inputs.