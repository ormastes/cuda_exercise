// test_synchronization_atomics.cu - Unit tests for synchronization and atomic operations
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace cg = cooperative_groups;

// Test fixture
class SynchronizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA initialization failed: " << cudaGetErrorString(err);
        }
    }

    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during test: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceReset();
    }
};

// ===============================
// Test Kernels
// ===============================

// Simple synchronization test kernel
__global__ void sync_test_kernel(int* data, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) {
        sdata[tid] = data[gid];
    }

    __syncthreads();

    // Simple reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && gid < N) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

// Atomic add test kernel
__global__ void atomic_add_test_kernel(int* counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(counter, 1);
    }
}

// Atomic CAS test kernel
__global__ void atomic_cas_test_kernel(int* value, int expected, int new_val, int* success_count) {
    int old_val = atomicCAS(value, expected, new_val);
    if (old_val == expected) {
        atomicAdd(success_count, 1);
    }
}

// Warp shuffle test kernel
__global__ void warp_shuffle_test_kernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;

    if (tid < N) {
        int value = data[tid];

        // Sum within warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }

        if (lane == 0) {
            data[tid] = value;
        }
    }
}

// Memory fence test kernels
__device__ volatile int test_flag = 0;
__device__ int test_data = 0;

__global__ void fence_writer_kernel(int value) {
    if (threadIdx.x == 0) {
        test_data = value;
        __threadfence();
        atomicExch((int*)&test_flag, 1);
    }
}

__global__ void fence_reader_kernel(int* result) {
    if (threadIdx.x == 0) {
        while (atomicAdd((int*)&test_flag, 0) == 0);
        __threadfence();
        *result = test_data;
    }
}

// Cooperative groups test kernel
__global__ void coop_groups_test_kernel(int* data, int N) {
    cg::thread_block block = cg::this_thread_block();
    int tid = block.thread_rank();
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) {
        data[gid] = gid * 2;
        block.sync();

        if (tid > 0 && gid > 0) {
            data[gid] += data[gid - 1];
        }
    }
}

// ===============================
// Tests
// ===============================

TEST_F(SynchronizationTest, BasicSyncthreads) {
    const int N = 512;
    const int blockSize = 128;
    const int gridSize = (N + blockSize - 1) / blockSize;

    std::vector<int> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = 1;
    }

    int* d_data;
    ASSERT_EQ(cudaMalloc(&d_data, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    size_t shared_size = blockSize * sizeof(int);
    sync_test_kernel<<<gridSize, blockSize, shared_size>>>(d_data, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int> h_result(gridSize);
    ASSERT_EQ(cudaMemcpy(h_result.data(), d_data, gridSize * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify reduction results
    int total = 0;
    for (int i = 0; i < gridSize; i++) {
        total += h_result[i];
    }

    EXPECT_EQ(total, N);

    cudaFree(d_data);
}

TEST_F(SynchronizationTest, AtomicAdd) {
    int* d_counter;
    ASSERT_EQ(cudaMalloc(&d_counter, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_counter, 0, sizeof(int)), cudaSuccess);

    const int threads = 256;
    const int blocks = 10;
    const int iterations = 100;

    atomic_add_test_kernel<<<blocks, threads>>>(d_counter, iterations);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_counter;
    ASSERT_EQ(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_counter, threads * blocks * iterations);

    cudaFree(d_counter);
}

TEST_F(SynchronizationTest, AtomicCAS) {
    int* d_value;
    int* d_success;

    ASSERT_EQ(cudaMalloc(&d_value, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_success, sizeof(int)), cudaSuccess);

    // Test CAS operation
    int initial_value = 42;
    int expected_value = 42;
    int new_value = 100;

    ASSERT_EQ(cudaMemcpy(d_value, &initial_value, sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_success, 0, sizeof(int)), cudaSuccess);

    // Launch 32 threads all trying to CAS the same value
    atomic_cas_test_kernel<<<1, 32>>>(d_value, expected_value, new_value, d_success);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_success;
    int h_final_value;
    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_final_value, d_value, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Exactly one thread should succeed in the CAS operation
    EXPECT_EQ(h_success, 1);
    // The value should be changed to new_value
    EXPECT_EQ(h_final_value, new_value);

    // Test CAS failure case
    ASSERT_EQ(cudaMemset(d_success, 0, sizeof(int)), cudaSuccess);

    // Try CAS with wrong expected value (should fail for all threads)
    atomic_cas_test_kernel<<<1, 32>>>(d_value, expected_value, 200, d_success);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&h_success, d_success, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_final_value, d_value, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // No thread should succeed since value is not what they expect
    EXPECT_EQ(h_success, 0);
    // Value should remain unchanged
    EXPECT_EQ(h_final_value, new_value);

    cudaFree(d_value);
    cudaFree(d_success);
}

TEST_F(SynchronizationTest, WarpShuffle) {
    const int warps = 4;
    const int N = warps * 32;

    std::vector<int> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = 1;
    }

    int* d_data;
    ASSERT_EQ(cudaMalloc(&d_data, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    warp_shuffle_test_kernel<<<1, N>>>(d_data, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int> h_result(N);
    ASSERT_EQ(cudaMemcpy(h_result.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Check warp reductions
    for (int w = 0; w < warps; w++) {
        EXPECT_EQ(h_result[w * 32], 32); // Each warp should sum to 32
    }

    cudaFree(d_data);
}

TEST_F(SynchronizationTest, MemoryFence) {
    int* d_result;
    ASSERT_EQ(cudaMalloc(&d_result, sizeof(int)), cudaSuccess);

    // Reset flag
    int zero = 0;
    ASSERT_EQ(cudaMemcpyToSymbol(test_flag, &zero, sizeof(int)), cudaSuccess);

    int test_value = 42;

    // Launch writer and reader
    fence_writer_kernel<<<1, 1>>>(test_value);
    fence_reader_kernel<<<1, 1>>>(d_result);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_result;
    ASSERT_EQ(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_EQ(h_result, test_value);

    cudaFree(d_result);
}

TEST_F(SynchronizationTest, CooperativeGroups) {
    const int N = 256;

    std::vector<int> h_data(N);
    int* d_data;

    ASSERT_EQ(cudaMalloc(&d_data, N * sizeof(int)), cudaSuccess);

    coop_groups_test_kernel<<<1, N>>>(d_data, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<int> h_result(N);
    ASSERT_EQ(cudaMemcpy(h_result.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify results
    EXPECT_EQ(h_result[0], 0);
    for (int i = 1; i < N; i++) {
        int expected = i * 2 + (i - 1) * 2;
        EXPECT_EQ(h_result[i], expected);
    }

    cudaFree(d_data);
}

TEST_F(SynchronizationTest, AtomicPerformance) {
    // Compare atomic vs non-atomic performance
    const int N = 1000000;
    int* d_atomic_counter;
    int* d_regular_array;

    ASSERT_EQ(cudaMalloc(&d_atomic_counter, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_regular_array, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_atomic_counter, 0, sizeof(int)), cudaSuccess);

    // Measure atomic operation time
    cudaEvent_t start, stop;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
    atomic_add_test_kernel<<<100, 256>>>(d_atomic_counter, 100);
    ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

    float atomic_time;
    ASSERT_EQ(cudaEventElapsedTime(&atomic_time, start, stop), cudaSuccess);

    int h_counter;
    ASSERT_EQ(cudaMemcpy(&h_counter, d_atomic_counter, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    std::cout << "\nAtomic operations: " << atomic_time << " ms, counter = " << h_counter << std::endl;

    // Verify correctness
    EXPECT_EQ(h_counter, 100 * 256 * 100);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_atomic_counter);
    cudaFree(d_regular_array);
}

// Test atomic operations on different data types
template<typename T>
__global__ void atomic_min_max_kernel(T* min_val, T* max_val, T* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicMin(min_val, data[tid]);
        atomicMax(max_val, data[tid]);
    }
}

TEST_F(SynchronizationTest, AtomicMinMax) {
    const int N = 1000;
    std::vector<int> h_data(N);

    // Generate random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 999);
    for (int i = 0; i < N; i++) {
        h_data[i] = dis(gen);
    }

    int* d_data;
    int* d_min;
    int* d_max;

    ASSERT_EQ(cudaMalloc(&d_data, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_min, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_max, sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    int init_min = INT_MAX;
    int init_max = INT_MIN;
    ASSERT_EQ(cudaMemcpy(d_min, &init_min, sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_max, &init_max, sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    atomic_min_max_kernel<<<(N + 255) / 256, 256>>>(d_min, d_max, d_data, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_min, h_max;
    ASSERT_EQ(cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify with CPU calculation
    int expected_min = *std::min_element(h_data.begin(), h_data.end());
    int expected_max = *std::max_element(h_data.begin(), h_data.end());

    EXPECT_EQ(h_min, expected_min);
    EXPECT_EQ(h_max, expected_max);

    cudaFree(d_data);
    cudaFree(d_min);
    cudaFree(d_max);
}

// Test race condition without synchronization
__global__ void race_condition_kernel(int* counter) {
    // Intentional race condition - no atomics
    (*counter)++;
}

__global__ void no_race_kernel(int* counter) {
    // Fixed with atomic
    atomicAdd(counter, 1);
}

TEST_F(SynchronizationTest, RaceConditionDetection) {
    int* d_counter1;
    int* d_counter2;

    ASSERT_EQ(cudaMalloc(&d_counter1, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_counter2, sizeof(int)), cudaSuccess);

    const int threads = 1000;
    const int blocks = 100;

    // Test with race condition
    ASSERT_EQ(cudaMemset(d_counter1, 0, sizeof(int)), cudaSuccess);
    race_condition_kernel<<<blocks, threads>>>(d_counter1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_counter1;
    ASSERT_EQ(cudaMemcpy(&h_counter1, d_counter1, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    // Test without race condition
    ASSERT_EQ(cudaMemset(d_counter2, 0, sizeof(int)), cudaSuccess);
    no_race_kernel<<<blocks, threads>>>(d_counter2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_counter2;
    ASSERT_EQ(cudaMemcpy(&h_counter2, d_counter2, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    std::cout << "\nRace condition test:" << std::endl;
    std::cout << "  With race: " << h_counter1 << " (expected: " << threads * blocks << ")" << std::endl;
    std::cout << "  Without race: " << h_counter2 << " (expected: " << threads * blocks << ")" << std::endl;

    // The atomic version should always be correct
    EXPECT_EQ(h_counter2, threads * blocks);

    // The race condition version is likely incorrect (but not guaranteed)
    // We can't assert on h_counter1 as the result is undefined

    cudaFree(d_counter1);
    cudaFree(d_counter2);
}