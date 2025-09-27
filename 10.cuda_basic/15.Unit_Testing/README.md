# Unit Testing for CUDA with GPU Testing Framework

This project demonstrates how to write unit tests for CUDA kernels using the GPU testing framework (gpu_gtest.h) that allows tests to run directly on the GPU.

## Project Structure

- `vector_add_2d.cu` - Optimized implementation of 2D vector operations with performance-focused kernels
- `vector_add_2d.h` - Header file with kernel declarations
- `test_vector_add_2d.cu` - Comprehensive unit tests using the GPU testing framework
- `CMakeLists.txt` - Build configuration with Google Test integration

## Implemented Kernels

### 1. `vectorAdd2D`
Optimized 2D vector addition kernel with strided memory access pattern (column-major in row-major storage).

```cuda
__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height)
```

**Features:**
- Uses column-major indexing (`x * height + y`) for testing strided memory patterns
- Incorporates `square()` device function for computation
- Demonstrates memory coalescing challenges

### 2. `reduceSum`
High-performance reduction kernel with grid-stride loop and optimized memory access.

```cuda
__global__ void reduceSum(const float* input, float* output, int N, int stride)
```

**Features:**
- **Grid-stride loop**: Each thread processes multiple elements for better memory throughput
- **Loop unrolling**: Compile-time optimizations for different block sizes
- **Warp-level reduction**: Exploits warp synchronization for the final 32 threads
- **Coalesced memory access**: Optimized for both regular and strided patterns
- **Multiple elements per thread**: Reduces kernel launch overhead

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

## CMake Configuration

### Root CMakeLists.txt Setup

The root CMakeLists.txt must be properly configured for CUDA and Google Test integration:

```cmake
......

# Testing support
enable_testing()

# FetchContent for downloading Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include Google Test's CMake utilities
include(CTest)
include(GoogleTest)
```

### Project-Specific CMakeLists.txt

The Unit Testing project's CMakeLists.txt configures the test executable:

```cmake
# Create test executable
add_executable(15_Unit_Testing_test
    test_vector_add_2d.cu
    vector_add_2d.cu
)

# Include directories
target_include_directories(15_Unit_Testing_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common  # For gpu_gtest.h
)

# Link libraries
target_link_libraries(15_Unit_Testing_test
    CUDA::cudart
    gtest
    gtest_main
)

# Set CUDA compilation flags
set_target_properties(15_Unit_Testing_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Register tests with CTest
gtest_discover_tests(15_Unit_Testing_test)
```

### Key Configuration Elements

| Configuration | Purpose | Required for |
|---------------|---------|--------------|
| `CMAKE_CUDA_ARCHITECTURES` | Target GPU compute capabilities | GPU kernel compilation |
| `CMAKE_CUDA_SEPARABLE_COMPILATION` | Enable device code linking | Multi-file CUDA projects |
| `FetchContent` | Download Google Test automatically | Unit testing framework |
| `gtest_discover_tests` | Register tests with CTest | Running tests via `ctest` |
| `CUDA::cudart` | CUDA runtime library | All CUDA applications |

### Build Configurations

Different build types for testing:

```bash
# Debug build (best for test development)
cmake -B build_debug -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --target 15_Unit_Testing_test

# Release build (for performance testing)
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target 15_Unit_Testing_test

# With Ninja for faster builds
cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Debug
ninja -C build 15_Unit_Testing_test
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
7. **Performance-Oriented Kernels**: Optimized implementations demonstrating best practices
8. **Memory Pattern Testing**: Kernels designed to test different memory access patterns

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

## Performance Optimizations Demonstrated

### Memory Access Patterns
The `vectorAdd2D` kernel intentionally uses a strided access pattern (column-major indexing) to demonstrate:
- Performance impact of non-coalesced memory access
- How strided patterns affect memory bandwidth
- Testing scenarios for memory-bound kernels

### Reduction Optimization Techniques
The `reduceSum` kernel showcases several optimization strategies:

1. **Grid-Stride Loops**: Allows kernels to process datasets larger than the grid size
2. **Loop Unrolling**: Template-based compile-time optimization for known block sizes
3. **Warp-Level Primitives**: Uses implicit warp synchronization for the last 32 threads
4. **Volatile Pointers**: Ensures proper memory visibility in the final reduction phase
5. **Multiple Elements per Thread**: Increases arithmetic intensity and reduces memory traffic

### Testing Performance Characteristics
The unit tests can verify:
- Correctness of optimized kernels
- Performance consistency across different data sizes
- Behavior with various launch configurations
- Edge cases with non-power-of-2 sizes

## Comparison with Part 14
This implementation incorporates the best practices from Part 14 (Code Inspection and Profiling):
- Optimized memory access patterns
- Performance-oriented kernel design
- Testing framework for verification
- Ready for profiling with Nsight tools