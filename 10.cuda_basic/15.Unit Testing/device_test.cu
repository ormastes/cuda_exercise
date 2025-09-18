#include "../../00.lib/gpu_gtest.h"
#include <cuda_runtime.h>

// ============================================================================
// Device functions to be tested
// ============================================================================

__device__ int add_integers(int a, int b) {
    return a + b;
}

__device__ int subtract_integers(int a, int b) {
    return a - b;
}

__device__ int multiply_integers(int a, int b) {
    return a * b;
}

__device__ int divide_integers(int a, int b) {
    return (b != 0) ? (a / b) : 0;
}

__device__ float compute_distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

__device__ float lerp_device(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ int clamp(int value, int min_val, int max_val) {
    return fminf(fmaxf(value, min_val), max_val);
}

__device__ bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

__device__ void vector_add(const float* a, const float* b, float* c, int idx) {
    c[idx] = a[idx] + b[idx];
}

__device__ float dot_product_element(const float* a, const float* b, int idx) {
    return a[idx] * b[idx];
}

__device__ void matrix_element_multiply(const float* matrix, float scalar, float* result, int idx) {
    result[idx] = matrix[idx] * scalar;
}

template<typename T>
__device__ void swap_device(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__device__ int factorial(int n) {
    if (n <= 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

__device__ float polynomial_eval(float x, const float* coeffs, int degree) {
    float result = coeffs[degree];
    for (int i = degree - 1; i >= 0; --i) {
        result = result * x + coeffs[i];
    }
    return result;
}

// ============================================================================
// Test Cases
// ============================================================================

GPU_TEST(DeviceFunctions, BasicArithmetic) {
    // Test add_integers
    int sum = add_integers(5, 3);
    GPU_EXPECT_EQ(sum, 8);

    // Test subtract_integers
    int diff = subtract_integers(10, 4);
    GPU_EXPECT_EQ(diff, 6);

    // Test multiply_integers
    int prod = multiply_integers(7, 6);
    GPU_EXPECT_EQ(prod, 42);

    // Test divide_integers
    int quot = divide_integers(20, 4);
    GPU_EXPECT_EQ(quot, 5);

    // Test division by zero handling
    int safe_div = divide_integers(10, 0);
    GPU_EXPECT_EQ(safe_div, 0);
}

GPU_TEST(DeviceFunctions, FloatingPointMath) {
    // Test compute_distance
    float dist = compute_distance(0.0f, 0.0f, 3.0f, 4.0f);
    GPU_EXPECT_NEAR(dist, 5.0f, 0.001f);

    dist = compute_distance(1.0f, 1.0f, 4.0f, 5.0f);
    GPU_EXPECT_NEAR(dist, 5.0f, 0.001f);

    // Test lerp_device
    float interpolated = lerp_device(0.0f, 10.0f, 0.5f);
    GPU_EXPECT_NEAR(interpolated, 5.0f, 0.001f);

    interpolated = lerp_device(2.0f, 8.0f, 0.25f);
    GPU_EXPECT_NEAR(interpolated, 3.5f, 0.001f);
}

GPU_TEST(DeviceFunctions, UtilityFunctions) {
    // Test clamp
    int clamped = clamp(15, 0, 10);
    GPU_EXPECT_EQ(clamped, 10);

    clamped = clamp(-5, 0, 10);
    GPU_EXPECT_EQ(clamped, 0);

    clamped = clamp(7, 0, 10);
    GPU_EXPECT_EQ(clamped, 7);

    // Test is_prime
    GPU_EXPECT_TRUE(is_prime(2));
    GPU_EXPECT_TRUE(is_prime(3));
    GPU_EXPECT_TRUE(is_prime(5));
    GPU_EXPECT_TRUE(is_prime(7));
    GPU_EXPECT_TRUE(is_prime(11));
    GPU_EXPECT_TRUE(is_prime(13));

    GPU_EXPECT_TRUE(!is_prime(1));
    GPU_EXPECT_TRUE(!is_prime(4));
    GPU_EXPECT_TRUE(!is_prime(6));
    GPU_EXPECT_TRUE(!is_prime(8));
    GPU_EXPECT_TRUE(!is_prime(9));
    GPU_EXPECT_TRUE(!is_prime(10));
}

GPU_TEST(DeviceFunctions, Factorial) {
    int fact = factorial(0);
    GPU_EXPECT_EQ(fact, 1);

    fact = factorial(1);
    GPU_EXPECT_EQ(fact, 1);

    fact = factorial(5);
    GPU_EXPECT_EQ(fact, 120);

    fact = factorial(6);
    GPU_EXPECT_EQ(fact, 720);
}

GPU_TEST(DeviceFunctions, TemplateSwap) {
    int a = 5, b = 10;
    swap_device(a, b);
    GPU_EXPECT_EQ(a, 10);
    GPU_EXPECT_EQ(b, 5);

    float x = 3.14f, y = 2.71f;
    swap_device(x, y);
    GPU_EXPECT_NEAR(x, 2.71f, 0.001f);
    GPU_EXPECT_NEAR(y, 3.14f, 0.001f);
}

GPU_TEST_CFG(DeviceFunctions, VectorOperations, 1, 256) {

    __shared__ float shared_a[256];
    __shared__ float shared_b[256];
    __shared__ float shared_c[256];

    int tid = threadIdx.x;

    // Initialize shared memory
    shared_a[tid] = tid * 1.0f;
    shared_b[tid] = tid * 2.0f;
    __syncthreads();

    // Test vector_add device function
    vector_add(shared_a, shared_b, shared_c, tid);
    __syncthreads();

    float expected = tid * 3.0f;
    GPU_EXPECT_NEAR(shared_c[tid], expected, 0.001f);

    // Test dot_product_element
    float dot_elem = dot_product_element(shared_a, shared_b, tid);
    expected = tid * tid * 2.0f;
    GPU_EXPECT_NEAR(dot_elem, expected, 0.001f);
}

GPU_TEST(DeviceFunctions, PolynomialEvaluation) {
    // Test polynomial: 2x^2 + 3x + 1
    float coeffs[3] = {1.0f, 3.0f, 2.0f};  // [constant, x, x^2]

    float result = polynomial_eval(0.0f, coeffs, 2);
    GPU_EXPECT_NEAR(result, 1.0f, 0.001f);

    result = polynomial_eval(1.0f, coeffs, 2);
    GPU_EXPECT_NEAR(result, 6.0f, 0.001f);

    result = polynomial_eval(2.0f, coeffs, 2);
    GPU_EXPECT_NEAR(result, 15.0f, 0.001f);

    result = polynomial_eval(-1.0f, coeffs, 2);
    GPU_EXPECT_NEAR(result, 0.0f, 0.001f);
}

GPU_TEST_CFG(DeviceFunctions, MatrixOperations, 4, 64) {

    __shared__ float matrix[256];
    __shared__ float result[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < 256) {
        matrix[idx] = idx * 0.5f;

        // Test matrix_element_multiply
        matrix_element_multiply(matrix, 2.0f, result, idx);

        float expected = idx * 1.0f;
        GPU_EXPECT_NEAR(result[idx], expected, 0.001f);
    }
}

// ============================================================================
// Complex Device Function Tests with Fixtures
// ============================================================================

class MathOperationsFixture : public ::testing::Test {
public:
    struct DeviceView {
        float* input_a;
        float* input_b;
        float* output;
        int* int_array;
        int size;
    };

protected:
    void SetUp() override {
        size = 1024;

        cudaMallocManaged(&d_view, sizeof(DeviceView));
        cudaMallocManaged(&d_view->input_a, size * sizeof(float));
        cudaMallocManaged(&d_view->input_b, size * sizeof(float));
        cudaMallocManaged(&d_view->output, size * sizeof(float));
        cudaMallocManaged(&d_view->int_array, size * sizeof(int));
        d_view->size = size;

        for (int i = 0; i < size; i++) {
            d_view->input_a[i] = i * 0.1f;
            d_view->input_b[i] = i * 0.2f;
            d_view->int_array[i] = i;
        }

        cudaDeviceSynchronize();
    }

    void TearDown() override {
        cudaFree(d_view->input_a);
        cudaFree(d_view->input_b);
        cudaFree(d_view->output);
        cudaFree(d_view->int_array);
        cudaFree(d_view);
    }

    const DeviceView* device_view() const { return d_view; }

    GpuLaunchCfg launch_cfg() const {
        return MakeLaunchCfg((size + 255) / 256, 256);
    }

private:
    DeviceView* d_view;
    int size;
};

GPU_TEST_F(MathOperationsFixture, VectorAddWithDeviceFunction) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < _ctx->size) {
        // Use the device function to add vectors
        vector_add(_ctx->input_a, _ctx->input_b, _ctx->output, idx);

        // Verify the result
        float expected = _ctx->input_a[idx] + _ctx->input_b[idx];
        GPU_EXPECT_NEAR(_ctx->output[idx], expected, 0.001f);
    }
}

GPU_TEST_F(MathOperationsFixture, ComplexMathOperations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < _ctx->size) {
        // Test lerp_device on array elements
        float t = (float)idx / _ctx->size;
        float interpolated = lerp_device(_ctx->input_a[idx], _ctx->input_b[idx], t);
        float expected = _ctx->input_a[idx] + t * (_ctx->input_b[idx] - _ctx->input_a[idx]);
        GPU_EXPECT_NEAR(interpolated, expected, 0.001f);

        // Test clamp on integer array
        int clamped = clamp(_ctx->int_array[idx], 100, 500);
        if (idx < 100) {
            GPU_EXPECT_EQ(clamped, 100);
        } else if (idx > 500) {
            GPU_EXPECT_EQ(clamped, 500);
        } else {
            GPU_EXPECT_EQ(clamped, idx);
        }
    }
}

GPU_TEST_F_CFG(MathOperationsFixture, PrimeNumberCheck, 4, 256) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < 100) {  // Check first 100 numbers
        bool prime = is_prime(idx);

        // Known primes under 100
        if (idx == 2 || idx == 3 || idx == 5 || idx == 7 || idx == 11 ||
            idx == 13 || idx == 17 || idx == 19 || idx == 23 || idx == 29 ||
            idx == 31 || idx == 37 || idx == 41 || idx == 43 || idx == 47 ||
            idx == 53 || idx == 59 || idx == 61 || idx == 67 || idx == 71 ||
            idx == 73 || idx == 79 || idx == 83 || idx == 89 || idx == 97) {
            GPU_EXPECT_TRUE(prime);
        } else if (idx > 1) {
            GPU_EXPECT_TRUE(!prime);
        }
    }
}

// ============================================================================
// Advanced Device Function Tests
// ============================================================================

__device__ float recursive_sum(const float* arr, int start, int end) {
    if (start >= end) return 0.0f;
    if (start == end - 1) return arr[start];

    int mid = (start + end) / 2;
    return recursive_sum(arr, start, mid) + recursive_sum(arr, mid, end);
}

__device__ void bubble_sort_device(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap_device(arr[j], arr[j + 1]);
            }
        }
    }
}

GPU_TEST(DeviceFunctions, RecursiveSum) {
    float arr[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    float sum = recursive_sum(arr, 0, 8);
    GPU_EXPECT_NEAR(sum, 36.0f, 0.001f);

    sum = recursive_sum(arr, 2, 6);
    GPU_EXPECT_NEAR(sum, 18.0f, 0.001f);  // 3+4+5+6
}

GPU_TEST(DeviceFunctions, DeviceSorting) {
    int arr[10] = {9, 3, 7, 1, 5, 8, 2, 6, 4, 0};

    bubble_sort_device(arr, 10);

    for (int i = 0; i < 10; i++) {
        GPU_EXPECT_EQ(arr[i], i);
    }
}

// ============================================================================
// Warp-level Device Functions
// ============================================================================

__device__ int warp_reduce_sum(int value) {
    unsigned mask = __activemask();

    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset);
    }

    return value;
}

__device__ int warp_broadcast(int value, int src_lane) {
    unsigned mask = __activemask();
    return __shfl_sync(mask, value, src_lane);
}

GPU_TEST_CFG(DeviceFunctions, WarpLevelFunctions, 1, 32) {

    int lane_id = threadIdx.x % 32;

    // Test warp reduce sum
    int sum = warp_reduce_sum(1);
    if (lane_id == 0) {
        GPU_EXPECT_EQ(sum, 32);
    }

    // Test warp broadcast
    int broadcast_value = (lane_id == 5) ? 42 : 0;
    int received = warp_broadcast(broadcast_value, 5);
    GPU_EXPECT_EQ(received, 42);
}

// ============================================================================
// Atomic Device Functions
// ============================================================================

__device__ void atomic_add_device(int* counter, int value) {
    atomicAdd(counter, value);
}

__device__ int atomic_max_device(int* address, int value) {
    return atomicMax(address, value);
}

GPU_TEST_CFG(DeviceFunctions, AtomicDeviceFunctions, 4, 256) {

    __shared__ int counter;
    __shared__ int max_value;

    if (threadIdx.x == 0) {
        counter = 0;
        max_value = 0;
    }
    __syncthreads();

    // Test atomic_add_device
    atomic_add_device(&counter, 1);
    __syncthreads();

    if (threadIdx.x == 0) {
        GPU_EXPECT_EQ(counter, blockDim.x);
    }

    // Test atomic_max_device
    int my_value = threadIdx.x;
    atomic_max_device(&max_value, my_value);
    __syncthreads();

    if (threadIdx.x == 0) {
        GPU_EXPECT_EQ(max_value, blockDim.x - 1);
    }
}