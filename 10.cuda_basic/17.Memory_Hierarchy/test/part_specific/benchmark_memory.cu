// benchmark_memory.cu - Performance benchmarks for memory hierarchy
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../../src/part_specific/vector_add_memory.cu"
#include "../../src/kernels/matrix_multiply.cu"
#include "cuda_utils.h"

class MemoryBenchmark {
private:
    const int N = 1 << 24;  // 16M elements
    const int ITERATIONS = 100;
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    CudaTimer timer;

public:
    MemoryBenchmark() {
        // Allocate memory
        size_t bytes = N * sizeof(float);
        h_A = new float[N];
        h_B = new float[N];
        h_C = new float[N];

        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        // Initialize data
        for (int i = 0; i < N; i++) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(i * 2);
        }

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    }

    ~MemoryBenchmark() {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void benchmark_global_memory() {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        // Warmup
        vector_add_global<<<grid, block>>>(d_A, d_B, d_C, N);
        CHECK_KERNEL_LAUNCH();

        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_global<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();

        float avg_time = timer.elapsed_ms() / ITERATIONS;
        float bandwidth = calculate_bandwidth_gb(3 * N * sizeof(float), avg_time);

        std::cout << "Global Memory Vector Add:\n";
        std::cout << "  Time: " << avg_time << " ms\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";
    }

    void benchmark_shared_memory() {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        size_t shared_size = 2 * block.x * sizeof(float);

        // Warmup
        vector_add_shared<<<grid, block, shared_size>>>(d_A, d_B, d_C, N);
        CHECK_KERNEL_LAUNCH();

        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_shared<<<grid, block, shared_size>>>(d_A, d_B, d_C, N);
        }
        timer.stop();

        float avg_time = timer.elapsed_ms() / ITERATIONS;
        float bandwidth = calculate_bandwidth_gb(3 * N * sizeof(float), avg_time);

        std::cout << "Shared Memory Vector Add:\n";
        std::cout << "  Time: " << avg_time << " ms\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";
    }

    void benchmark_coalescing() {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        std::cout << "Memory Coalescing Comparison:\n";

        // Coalesced access (stride = 1)
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_global<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float coalesced_time = timer.elapsed_ms() / ITERATIONS;
        float coalesced_bw = calculate_bandwidth_gb(3 * N * sizeof(float), coalesced_time);

        // Strided access (stride = 32)
        int stride = 32;
        dim3 strided_grid((N/stride + block.x - 1) / block.x);

        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_strided<<<strided_grid, block>>>(d_A, d_B, d_C, N, stride);
        }
        timer.stop();
        float strided_time = timer.elapsed_ms() / ITERATIONS;
        float strided_bw = calculate_bandwidth_gb(3 * N/stride * sizeof(float), strided_time);

        std::cout << "  Coalesced: " << coalesced_time << " ms, " << coalesced_bw << " GB/s\n";
        std::cout << "  Strided:   " << strided_time << " ms, " << strided_bw << " GB/s\n";
        std::cout << "  Speedup: " << strided_time / coalesced_time << "x\n\n";
    }

    void benchmark_bank_conflicts() {
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        std::cout << "Bank Conflict Analysis:\n";

        // With bank conflicts
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_bank_conflict<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float conflict_time = timer.elapsed_ms() / ITERATIONS;

        // Without bank conflicts
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            vector_add_bank_conflict_free<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float no_conflict_time = timer.elapsed_ms() / ITERATIONS;

        std::cout << "  With conflicts:    " << conflict_time << " ms\n";
        std::cout << "  Without conflicts: " << no_conflict_time << " ms\n";
        std::cout << "  Speedup: " << conflict_time / no_conflict_time << "x\n\n";
    }
};

class MatrixBenchmark {
private:
    const int N = 1024;  // Matrix size
    const int ITERATIONS = 10;
    float *d_A, *d_B, *d_C;
    CudaTimer timer;

public:
    MatrixBenchmark() {
        size_t bytes = N * N * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        // Initialize with random data
        std::vector<float> h_A(N * N), h_B(N * N);
        for (int i = 0; i < N * N; i++) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~MatrixBenchmark() {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void benchmark_all() {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        std::cout << "Matrix Multiplication Performance (N=" << N << "):\n\n";

        // Naive implementation
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float naive_time = timer.elapsed_ms() / ITERATIONS;
        float naive_gflops = calculate_gflops(2.0 * N * N * N, naive_time);
        std::cout << "Naive: " << naive_time << " ms, " << naive_gflops << " GFLOPS\n";

        // Coalesced access
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            matmul_coalesced<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float coalesced_time = timer.elapsed_ms() / ITERATIONS;
        float coalesced_gflops = calculate_gflops(2.0 * N * N * N, coalesced_time);
        std::cout << "Coalesced: " << coalesced_time << " ms, " << coalesced_gflops << " GFLOPS\n";

        // Shared memory tiled
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            matmul_shared<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float shared_time = timer.elapsed_ms() / ITERATIONS;
        float shared_gflops = calculate_gflops(2.0 * N * N * N, shared_time);
        std::cout << "Shared: " << shared_time << " ms, " << shared_gflops << " GFLOPS\n";

        // Bank conflict free
        timer.start();
        for (int i = 0; i < ITERATIONS; i++) {
            matmul_shared_bank_conflict_free<<<grid, block>>>(d_A, d_B, d_C, N);
        }
        timer.stop();
        float bank_free_time = timer.elapsed_ms() / ITERATIONS;
        float bank_free_gflops = calculate_gflops(2.0 * N * N * N, bank_free_time);
        std::cout << "Bank-Free: " << bank_free_time << " ms, " << bank_free_gflops << " GFLOPS\n";

        std::cout << "\nSpeedups over naive:\n";
        std::cout << "  Coalesced: " << naive_time / coalesced_time << "x\n";
        std::cout << "  Shared:    " << naive_time / shared_time << "x\n";
        std::cout << "  Bank-Free: " << naive_time / bank_free_time << "x\n";
    }
};

int main() {
    print_device_info();

    std::cout << "\n=== Vector Addition Memory Hierarchy Benchmarks ===\n\n";
    MemoryBenchmark mem_bench;
    mem_bench.benchmark_global_memory();
    mem_bench.benchmark_shared_memory();
    mem_bench.benchmark_coalescing();
    mem_bench.benchmark_bank_conflicts();

    std::cout << "\n=== Matrix Multiplication Memory Optimizations ===\n\n";
    MatrixBenchmark mat_bench;
    mat_bench.benchmark_all();

    return 0;
}