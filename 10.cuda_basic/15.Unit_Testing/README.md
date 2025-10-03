# 🧪 Part 15: Unit Testing for CUDA

**Goal**: Implement comprehensive testing for CUDA kernels using custom GPU testing framework with GTest integration.

This project demonstrates how to write unit tests for CUDA kernels using the GPU testing framework (gpu_gtest.h) that allows tests to run directly on the GPU.

## Project Structure

- `vector_add_2d.cu` - Optimized implementation of 2D vector operations with performance-focused kernels
- `vector_add_2d.h` - Header file with kernel declarations
- `test_vector_add_2d.cu` - Comprehensive unit tests using the GPU testing framework (direct inclusion method)
- `test_vector_add_2d_with_lib.cu` - Unit tests using library-based testing approach
- `CMakeLists.txt` - Build configuration with Google Test integration for both testing approaches

## Implemented Kernels

### 1. `vector_add_2d`
Optimized 2D vector addition kernel with strided memory access pattern (column-major in row-major storage).

```cuda
__global__ void vector_add_2d(const float* A, const float* B, float* C, int width, int height)
```

**Features:**
- Uses column-major indexing (`x * height + y`) for testing strided memory patterns
- Incorporates `square()` device function for computation
- Demonstrates memory coalescing challenges

### 2. `reduce_sum`
High-performance reduction kernel with grid-stride loop and optimized memory access.

```cuda
__global__ void reduce_sum(const float* input, float* output, int N, int stride)
```

**Features:**
- **Grid-stride loop**: Each thread processes multiple elements for better memory throughput
- **Loop unrolling**: Compile-time optimizations for different block sizes
- **Warp-level reduction**: Exploits warp synchronization for the final 32 threads
- **Coalesced memory access**: Optimized for both regular and strided patterns
- **Multiple elements per thread**: Reduces kernel launch overhead

## Testing Approaches for CUDA Device Functions

This project demonstrates two methods for testing CUDA device functions and kernels:

### Method 1: Direct .cu File Inclusion (Recommended Default)

**File:** `test_vector_add_2d.cu`

This approach directly includes the CUDA implementation file in the test file, allowing access to all device functions, including `__device__` functions that are not normally accessible from external compilation units.

**Advantages:**
- Can test `__device__` functions directly
- No need to create separate libraries
- Simpler build configuration
- Full access to implementation details for white-box testing
- Better for unit testing individual device functions

**Example:**
```cuda
// test_vector_add_2d.cu
#include "vector_add_2d.cu"  // Direct inclusion of implementation

GPU_TEST(DeviceFunctionTest, TestSquare) {
    // Can directly test __device__ functions
    float result = square(3.0f);
    GPU_EXPECT_NEAR(result, 9.0f, 1e-5f);
}
```

**CMake Configuration:**
```cmake
add_executable(${PROJECT_NAME}_test
    test_vector_add_2d.cu
    # Note: Don't include vector_add_2d.cu here since it's #included
)
```

### Method 2: Library-Based Testing with Inline Functions

**File:** `test_vector_add_2d_with_lib.cu`

This approach creates a library from the CUDA code and tests it through its public interface. Device functions must be marked as `__device__ __inline__` in the header file to be accessible.

**Advantages:**
- Tests the actual library interface
- Better for integration testing
- Mimics real-world usage patterns
- Enforces proper API design

**Limitations:**
- Cannot test private `__device__` functions unless they're inline
- Requires more complex build setup
- Need to manage library dependencies

**Example:**
```cuda
// vector_add_2d.h - Functions must be inline
__device__ __inline__ float square(float x) {
    return x * x;
}

// test_vector_add_2d_with_lib.cu
#include "vector_add_2d.h"  // Include header only

GPU_TEST(LibraryTest, TestSquareViaLib) {
    // Test inline device functions from header
    float result = square(3.0f);
    GPU_EXPECT_NEAR(result, 9.0f, 1e-5f);
}
```

**CMake Configuration:**
```cmake
# Create library
add_library(vector_add_lib STATIC
    vector_add_2d.cu
)

# Create test executable
add_executable(${PROJECT_NAME}_test_with_lib
    test_vector_add_2d_with_lib.cu
)

# Link test with library
target_link_libraries(${PROJECT_NAME}_test_with_lib
    PRIVATE
    vector_add_lib
    GTest::gtest_main
)
```

### Recommendation

**Use Method 1 (Direct Inclusion) as the default** for most unit testing scenarios because:
1. It provides complete access to all device functions for thorough testing
2. Simpler to set up and maintain
3. Better suited for unit testing individual components
4. No need to expose internal functions in headers

**Use Method 2 (Library-Based) when:**
- Testing library APIs as they will be used by consumers
- Performing integration tests
- Working with pre-built libraries
- Enforcing API boundaries

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

### 4. GPU_TEST_G - Generator-Based Test
Test that uses generator syntax for parameterized testing on GPU.

```cuda
GPU_TEST_G(GpuGeneratorTest, square) {
    float a = GPU_GENERATOR(1.0f, 2.0f, 3.0f, 4.0f);
    float expected_square = GPU_GENERATOR(1.0f, 4.0f, 9.0f, 16.0f);
    GPU_USE_GENERATOR(ALIGNED);  // 4 iterations with aligned values

    float result = square(a);
    GPU_EXPECT_NEAR(result, expected_square, 1e-5f);
}
```

### 5. Host-Side Generator Tests (TEST_G)
Using gtest-parameterized-lib for host-side parameterized tests with intuitive generator syntax.

```cpp
class HostGeneratorTest : public ::gtest_generator::TestWithGenerator {};

TEST_G(HostGeneratorTest, VectorAdd2D) {
    // Generate test parameters
    int width = GENERATOR(16, 32, 64);
    int height = GENERATOR(16, 32, 64);
    float a_value = GENERATOR(1.0f, 2.0f, 3.0f);
    float b_value = GENERATOR(1.0f, 2.0f, 3.0f);
    USE_GENERATOR(ALIGNED);  // Use aligned mode for fewer test runs

    // Test implementation...
}
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

The Unit Testing project's CMakeLists.txt configures both testing approaches:

#### Method 1: Direct Inclusion Test (Default)
```cmake
# Create test executable with direct .cu file inclusion
add_executable(${PROJECT_NAME}_test
    test_vector_add_2d.cu
    # Note: vector_add_2d.cu is #included in test file, not compiled separately
)

# Link test with testing frameworks
target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib  # Custom CUDA utilities library
)

# Register tests with CTest
gtest_discover_tests(${PROJECT_NAME}_test)
```

#### Method 2: Library-Based Test
```cmake
# Create test executable with library approach
add_executable(${PROJECT_NAME}_test_with_library
    test_vector_add_2d_with_lib.cu
)

# Link test with implementation library and testing frameworks
target_link_libraries(${PROJECT_NAME}_test_with_library
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
    # Would link to vector_add_lib if using separate library
)

gtest_discover_tests(${PROJECT_NAME}_test_with_library)
```

#### Complete CMakeLists.txt Example
```cmake
cmake_minimum_required(VERSION 3.18)
project(15_Unit_Testing CUDA CXX)

# Method 1: Direct inclusion test (recommended default)
add_executable(${PROJECT_NAME}_test
    test_vector_add_2d.cu
)

target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
)

gtest_discover_tests(${PROJECT_NAME}_test)

# Method 2: Library-based test
add_executable(${PROJECT_NAME}_test_with_library
    test_vector_add_2d_with_lib.cu
)

target_link_libraries(${PROJECT_NAME}_test_with_library
    PRIVATE
    GTest::gtest_main
    GTestCudaGenerator
    CudaCustomLib
)

gtest_discover_tests(${PROJECT_NAME}_test_with_library)
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

### Method 1: Direct Inclusion Tests (Default)
```bash
# List all tests
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test --gtest_list_tests

# Run all tests
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test

# Run specific test
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test --gtest_filter="SimpleDeviceTest.*"
```

### Method 2: Library-Based Tests
```bash
# List all tests
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test_with_library --gtest_list_tests

# Run all tests
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test_with_library

# Run specific test
./build/10.cuda_basic/15.Unit_Testing/15_Unit_Testing_test_with_library --gtest_filter="LibraryTest.*"
```

### Running Both Test Suites with CTest
```bash
# Run all tests from build directory
cd build
ctest --test-dir 10.cuda_basic/15.Unit_Testing

# Run with verbose output
ctest --test-dir 10.cuda_basic/15.Unit_Testing --verbose

# Run only direct inclusion tests
ctest -R "15_Unit_Testing_test$"

# Run only library-based tests
ctest -R "15_Unit_Testing_test_with_library"
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

## Test Macros Reference

### GPU Test Macros
| Macro | Purpose | Launch Config |
|-------|---------|---------------|
| `GPU_TEST(Suite, Name)` | Simple device test | Default <<<1,1>>> |
| `GPU_TEST_CFG(Suite, Name, grid, block, ...)` | Test with explicit config | User-specified |
| `GPU_TEST_F(Fixture, Name)` | Fixture-based test | From fixture's launch_cfg() |
| `GPU_TEST_G(Suite, Name)` | Generator-based test | Default <<<1,1>>> |
| `GPU_TEST_G_CFG(Suite, Name, grid, block)` | Generator test with config | User-specified |

### Host Test Macros (with gtest-parameterized-lib)
| Macro | Purpose | Description |
|-------|---------|-------------|
| `TEST_G(Fixture, Name)` | Generator-based host test | Uses GENERATOR() for parameterization |
| `GENERATOR(...)` | Define test values | Creates parameter combinations |
| `USE_GENERATOR()` | Activate generators | Must be called after all GENERATOR() calls |
| `USE_GENERATOR(ALIGNED)` | Aligned mode | Reduces test count by parallel iteration |

## Test Assertions Available in Device Code

- `GPU_EXPECT_TRUE(condition)` - Check if condition is true
- `GPU_EXPECT_EQ(a, b)` - Check if values are equal
- `GPU_EXPECT_NEAR(a, b, tolerance)` - Check if values are within tolerance

## Key Features

1. **Direct GPU Testing**: Tests run directly on the GPU, allowing verification of device functions and kernels
2. **Generator-Based Testing**: Intuitive GENERATOR() syntax for both GPU and host tests using [gtest-parameterized-lib](https://github.com/ormastes/gtest-parameterized-lib)
3. **Fixture Support**: Setup and teardown device memory with fixture classes
4. **Parameterized Testing**: Run the same test with different input values using generator syntax
5. **Custom Launch Configurations**: Control grid and block dimensions for tests
6. **Integration with GTest**: Seamless integration with Google Test framework
7. **Host Integration Tests**: Traditional CPU-side tests for kernel verification with generator support
8. **Performance-Oriented Kernels**: Optimized implementations demonstrating best practices
9. **Memory Pattern Testing**: Kernels designed to test different memory access patterns
10. **Sampling Modes**: Support for FULL (Cartesian product) and ALIGNED (parallel iteration) test generation

## Best Practices

1. **Choose the right testing approach**:
   - **Direct inclusion (Method 1)** for unit testing device functions
   - **Library-based (Method 2)** for integration testing and API validation
   - Use direct inclusion as the default for new projects

2. **Use appropriate test type**:
   - GPU_TEST for simple device function tests
   - GPU_TEST_CFG when you need specific thread configurations
   - GPU_TEST_F for tests requiring complex setup/teardown
   - GPU_TEST_G for testing with multiple input values using generators

3. **Memory management**:
   - Always check CUDA error codes
   - Free allocated memory in teardown
   - Use RAII patterns where possible

4. **Test organization**:
   - Group related tests in test suites
   - Use descriptive test names
   - Test both success and edge cases

5. **Synchronization**:
   - Remember that GPU tests are asynchronous
   - Use proper synchronization for host tests
   - Check both launch and execution errors

6. **Testing strategy**:
   - Start with direct inclusion for complete coverage
   - Add library-based tests for public API validation
   - Use both approaches when testing complex libraries

## Performance Optimizations Demonstrated

### Memory Access Patterns
The `vector_add_2d` kernel intentionally uses a strided access pattern (column-major indexing) to demonstrate:
- Performance impact of non-coalesced memory access
- How strided patterns affect memory bandwidth
- Testing scenarios for memory-bound kernels

### Reduction Optimization Techniques
The `reduce_sum` kernel showcases several optimization strategies:

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

## Summary: Testing Approaches

This unit testing example demonstrates two complementary approaches for testing CUDA code:

1. **Direct .cu File Inclusion (Default Method)**: Best for unit testing with full access to device functions
2. **Library-Based Testing**: Best for integration testing and API validation

**For future examples in this tutorial series, we will use the direct inclusion method as the default** because it provides:
- Complete testability of all device functions
- Simpler setup and maintenance
- Better educational value for understanding CUDA internals
- No need to modify implementation code for testing

The library-based approach remains available for scenarios requiring true black-box testing or when working with pre-compiled CUDA libraries.