# Unit Testing for CUDA with GPU Testing Framework

This project demonstrates how to write unit tests for CUDA kernels using the GPU testing framework (gpu_gtest.h) that allows tests to run directly on the GPU.

## Project Structure

- `vector_add_2d.cu` - Implementation of 2D vector operations kernels
- `vector_add_2d.h` - Header file with kernel declarations and helper functions
- `test_vector_add_2d.cu` - Unit tests using the GPU testing framework
- `CMakeLists.txt` - Build configuration

## Implemented Kernels

### 1. `vector_add_2d`
Simple 2D vector addition kernel that adds two matrices element-wise.

```cuda
__global__ void vector_add_2d(const float* a, const float* b, float* c, int width, int height)
```

### 2. `reduce_sum_2d`
2D reduction kernel that sums all elements in a matrix using shared memory and atomic operations.

```cuda
__global__ void reduce_sum_2d(const float* input, float* output, int width, int height)
```

## Test Types Demonstrated

The project showcases 4 different GPU test macros:

### 1. GPU_TEST - Simple Device Test
Basic test that runs on the GPU with default launch configuration (<<<1,1>>>).

```cuda
GPU_TEST(SimpleDeviceTest, ComputeSum) {
    float result = compute_sum(3.0f, 4.0f);
    GPU_EXPECT_NEAR(result, 7.0f, 1e-5f);
}
```

### 2. GPU_TEST_CFG - Test with Custom Configuration
Test with explicit launch configuration (grid and block dimensions).

```cuda
GPU_TEST_CFG(ConfiguredTest, ParallelSum, dim3(1), dim3(32)) {
    // Test code using 32 threads
    int tid = threadIdx.x;
    if (tid < 10) {
        float value = compute_sum(float(tid), float(tid * 2));
        GPU_EXPECT_NEAR(value, float(tid * 3), 1e-5f);
    }
}
```

### 3. GPU_TEST_F - Fixture-based Test
Test using a fixture class that provides setup/teardown and device context.

```cuda
struct ReductionFixture : ::testing::Test {
    struct DeviceView {
        float* data;
        float* result;
        int size;
    };

    // Setup device memory and context
    void SetUp() override { /* ... */ }
    void TearDown() override { /* ... */ }
    const DeviceView* device_view() const { return d_view; }
    GpuLaunchCfg launch_cfg() const { /* ... */ }
};

GPU_TEST_F(ReductionFixture, SumElements) {
    // Access fixture data via _ctx pointer
    int tid = threadIdx.x;
    if (tid < _ctx->size) {
        float value = _ctx->data[tid];
        GPU_EXPECT_NEAR(value, 1.0f, 1e-5f);
    }
}
```

### 4. GPU_TEST_P - Parameterized Test
Test that runs multiple times with different parameter values.

```cuda
GPU_TEST_P(ParameterizedTest, AddValues) {
    float param = _param;  // Access parameter
    float result = compute_sum(param, param);
    GPU_EXPECT_NEAR(result, param * 2.0f, 1e-5f);
}

GPU_INSTANTIATE_TEST_SUITE_P(Values, ParameterizedTest, AddValues,
    ::testing::Values(1.0f, 2.0f, 3.0f, 5.0f, 10.0f));
```

## Building

The project is built as part of the parent CUDA exercise project:

```bash
# From the root cuda_exercise directory
cmake -B build -S .
cmake --build build --target 15_Unit_Testing_test
```

## Running Tests

```bash
# List all tests
./build/10.cuda_basic/15.Unit\ Testing/15_Unit_Testing_test --gtest_list_tests

# Run all tests
./build/10.cuda_basic/15.Unit\ Testing/15_Unit_Testing_test

# Run specific test
./build/10.cuda_basic/15.Unit\ Testing/15_Unit_Testing_test --gtest_filter="SimpleDeviceTest.*"
```

## Test Output

```
[==========] Running 11 tests from 6 test suites.
[----------] 1 test from SimpleDeviceTest
[ RUN      ] SimpleDeviceTest.ComputeSum
[       OK ] SimpleDeviceTest.ComputeSum (5 ms)
[----------] 1 test from ConfiguredTest
[ RUN      ] ConfiguredTest.ParallelSum
[       OK ] ConfiguredTest.ParallelSum (0 ms)
[----------] 1 test from ReductionFixture
[ RUN      ] ReductionFixture.SumElements
[       OK ] ReductionFixture.SumElements (0 ms)
[----------] 5 tests from Values/ParameterizedTest_AddValues_TestBase
[ RUN      ] Values/ParameterizedTest_AddValues_TestBase.Test/0
[       OK ] Values/ParameterizedTest_AddValues_TestBase.Test/0 (0 ms)
...
[----------] 2 tests from HostIntegrationTest
[ RUN      ] HostIntegrationTest.VectorAdd2D
[       OK ] HostIntegrationTest.VectorAdd2D (1 ms)
[ RUN      ] HostIntegrationTest.ReduceSum2D
[       OK ] HostIntegrationTest.ReduceSum2D (0 ms)
[==========] 11 tests from 6 test suites ran. (10 ms total)
[  PASSED  ] 11 tests.
```

## GPU Test Macros Reference

| Macro | Purpose | Launch Config |
|-------|---------|---------------|
| `GPU_TEST(Suite, Name)` | Simple device test | Default <<<1,1>>> |
| `GPU_TEST_CFG(Suite, Name, grid, block, ...)` | Test with explicit config | User-specified |
| `GPU_TEST_F(Fixture, Name)` | Fixture-based test | From fixture's launch_cfg() |
| `GPU_TEST_P(Suite, Name)` | Parameterized test | Default <<<1,1>>> |

## Test Assertions Available in Device Code

- `GPU_EXPECT_TRUE(condition)` - Check if condition is true
- `GPU_EXPECT_EQ(a, b)` - Check if values are equal
- `GPU_EXPECT_NEAR(a, b, tolerance)` - Check if values are within tolerance

## Key Features

1. **Direct GPU Testing**: Tests run directly on the GPU, allowing verification of device functions and kernels
2. **Fixture Support**: Setup and teardown device memory with fixture classes
3. **Parameterized Testing**: Run the same test with different input values
4. **Custom Launch Configurations**: Control grid and block dimensions for tests
5. **Integration with GTest**: Seamless integration with Google Test framework
6. **Host Integration Tests**: Traditional CPU-side tests for kernel verification

## Best Practices

1. **Use appropriate test type**:
   - GPU_TEST for simple device function tests
   - GPU_TEST_CFG when you need specific thread configurations
   - GPU_TEST_F for tests requiring complex setup/teardown
   - GPU_TEST_P for testing with multiple input values

2. **Memory management**:
   - Always check CUDA error codes
   - Free allocated memory in teardown
   - Use RAII patterns where possible

3. **Test organization**:
   - Group related tests in test suites
   - Use descriptive test names
   - Test both success and edge cases

4. **Synchronization**:
   - Remember that GPU tests are asynchronous
   - Use proper synchronization for host tests
   - Check both launch and execution errors