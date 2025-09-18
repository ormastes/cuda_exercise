#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void simpleKernel(int* result, int value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    result[idx] = value + idx;
}

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

TEST(HostLevel, DeviceDetection) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    EXPECT_EQ(error, cudaSuccess);
    EXPECT_GT(deviceCount, 0) << "No CUDA devices found";

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        EXPECT_EQ(error, cudaSuccess);
        EXPECT_GT(prop.major, 0);

        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
}

TEST(HostLevel, MemoryOperations) {
    const size_t size = 1024 * sizeof(float);
    float* d_data = nullptr;

    EXPECT_EQ(cudaMalloc(&d_data, size), cudaSuccess);
    EXPECT_NE(d_data, nullptr);

    EXPECT_EQ(cudaMemset(d_data, 0, size), cudaSuccess);

    float* h_data = new float[1024];
    for (int i = 0; i < 1024; i++) {
        h_data[i] = static_cast<float>(i);
    }

    EXPECT_EQ(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice), cudaSuccess);

    float* h_result = new float[1024];
    EXPECT_EQ(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < 1024; i++) {
        EXPECT_FLOAT_EQ(h_result[i], h_data[i]);
    }

    delete[] h_data;
    delete[] h_result;

    EXPECT_EQ(cudaFree(d_data), cudaSuccess);
}

TEST(HostLevel, KernelLaunch) {
    const int numElements = 256;
    int* d_result;
    std::vector<int> h_result(numElements);

    ASSERT_EQ(cudaMalloc(&d_result, numElements * sizeof(int)), cudaSuccess);

    simpleKernel<<<1, numElements>>>(d_result, 100);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

    ASSERT_EQ(cudaMemcpy(h_result.data(), d_result, numElements * sizeof(int),
                         cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < numElements; i++) {
        EXPECT_EQ(h_result[i], 100 + i) << "Mismatch at index " << i;
    }

    cudaFree(d_result);
}

TEST(HostLevel, VectorAddition) {
    const int n = 10000;
    const size_t size = n * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    ASSERT_EQ(cudaMalloc(&d_a, size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, size), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, size), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice), cudaSuccess);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost), cudaSuccess);

    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(h_c[i], h_a[i] + h_b[i]) << "Mismatch at index " << i;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

class CudaMemoryRAII {
    void* ptr = nullptr;
    size_t size = 0;
public:
    CudaMemoryRAII(size_t s) : size(s) {
        cudaMalloc(&ptr, size);
    }
    ~CudaMemoryRAII() {
        if (ptr) cudaFree(ptr);
    }
    void* get() { return ptr; }
    operator bool() const { return ptr != nullptr; }
};

TEST(HostLevel, RAIIMemoryPattern) {
    const size_t size = 1024 * sizeof(float);

    {
        CudaMemoryRAII mem(size);
        ASSERT_TRUE(mem);
        ASSERT_NE(mem.get(), nullptr);

        EXPECT_EQ(cudaMemset(mem.get(), 0, size), cudaSuccess);
    }

    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(error, cudaSuccess);
}

TEST(HostLevel, ErrorHandling) {
    void* ptr = nullptr;

    cudaError_t error = cudaMalloc(&ptr, SIZE_MAX);
    EXPECT_NE(error, cudaSuccess);
    EXPECT_EQ(ptr, nullptr);

    cudaGetLastError();

    simpleKernel<<<0, 256>>>(nullptr, 0);
    error = cudaGetLastError();
    EXPECT_NE(error, cudaSuccess);
}