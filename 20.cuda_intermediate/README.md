# 🚀 Intermediate CUDA Programming Tutorial

> **Note**: This section covers intermediate and advanced CUDA topics including advanced memory management, streams, and NVIDIA libraries.
>
> **Prerequisites**: Complete [10.cuda_basic](../10.cuda_basic/README.md) (Parts 1-9) first.

---

## 🔒 Part 21: Synchronization and Atomics

**Goal**: Advanced thread synchronization, atomic operations, and lock-free algorithms.

- **21.1** Thread Synchronization Primitives
- **21.2** Atomic Operations and Memory Ordering
- **21.3** Lock-Free Data Structures
- **21.4** Barriers and Memory Fences
- **21.5** CUB Library for Reductions
- **21.6** Avoiding Deadlocks and Race Conditions

📄 *Example Code:* `matrix_multiply_atomic.cu` (uses atomics for accumulation), `parallel_reduction.cu`

---

## 🧮 Part 22: Streams and Asynchronous Execution

**Goal**: Master CUDA streams for overlapping computation and data transfer.

- **22.1** Understanding CUDA Streams
- **22.2** Creating and Managing Streams
- **22.3** Async Memory Copy and Kernel Execution
- **22.4** Stream Priorities
- **22.5** Events for Synchronization and Timing
- **22.6** Multi-Stream Patterns
- **22.7** CUDA Graph API

📄 *Example Code:* `matrix_multiply_streams.cu` (parallel matrix operations with streams), `pipeline_compute.cu`

---

## 💾 Part 23: Shared Memory

**Goal**: Master shared memory usage for high-performance CUDA kernels through efficient data sharing and memory access patterns.

- **23.1** Shared Memory Fundamentals
- **23.2** Bank Conflicts and Padding
- **23.3** Classic Patterns (Tiling, Reduction, Stencil)
- **23.4** Double Buffering Techniques
- **23.5** Shared Memory Atomics
- **23.6** Performance Optimization
- **23.7** Advanced Techniques

📄 *Example Code:* `matrix_transpose.cu`, `convolution_1d.cu`, `reduction.cu`, `stencil_1d.cu`

---

## 🎯 Part 24: Memory Coalescing and Bank Conflicts

**Goal**: Optimize memory access patterns for maximum bandwidth utilization and minimal conflicts.

- **24.1** Memory Coalescing Fundamentals
- **24.2** Global Memory Access Patterns
- **24.3** Structure of Arrays vs Array of Structures
- **24.4** Shared Memory Bank Conflicts
- **24.5** Alignment and Padding Strategies
- **24.6** Vectorized Memory Access
- **24.7** Performance Analysis and Profiling

📄 *Example Code:* `coalescing_demo.cu`, `bank_conflicts.cu`, `aos_vs_soa.cu`, `memory_patterns.cu`

---

## 🌀 Part 25: Dynamic Parallelism

**Goal**: Master dynamic parallelism to launch kernels from within kernels, enabling recursive algorithms and adaptive workloads.

- **25.1** Dynamic Parallelism Fundamentals
- **25.2** Device Runtime API
- **25.3** Recursive Algorithms (Quicksort, Tree Traversal)
- **25.4** Adaptive Algorithms (Mesh Refinement, Integration)
- **25.5** Nested Parallelism Patterns
- **25.6** Memory Management in Device Code
- **25.7** Performance Considerations and Best Practices

📄 *Example Code:* `quicksort.cu`, `tree_traversal.cu`, `adaptive_mesh.cu`, `recursive_matmul.cu`

---

## 🧩 Part 26: Using CUDA Libraries

**Goal**: Leverage NVIDIA's optimized libraries for common operations.

### **26.1 cuBLAS - Linear Algebra**
- Basic Operations (Level 1, 2, 3 BLAS)
- Matrix Multiplication with `cublasSgemm`
- Batched Operations
- Mixed Precision with Tensor Cores

📄 *Example Code:* `matrix_multiply_cublas.cu` (comparing custom kernel vs cuBLAS), `backprop_layer.cu` (neural network layer forward/backward)

### **26.2 cuFFT - Fast Fourier Transforms**
- 1D, 2D, and 3D Transforms
- Real and Complex Transforms
- Batched FFTs
- Performance Optimization

📄 *Example Code:* `cufft_convolution.cu`, `cufft_image_filter.cu` (FFT-based image processing)

### **26.3 cuRAND - Random Number Generation**
- Pseudo-Random Generators
- Quasi-Random Generators
- Distribution Functions
- Monte Carlo Simulations

📄 *Example Code:* `backprop_init.cu` (weight initialization for neural networks), `curand_monte_carlo.cu`

### **26.4 cuSPARSE - Sparse Matrix Operations**
- Sparse Matrix Formats (CSR, COO)
- Sparse Matrix-Vector Multiplication
- Format Conversions

📄 *Example Code:* `sparse_matmul.cu` (sparse matrix multiplication), `sparse_gradient.cu` (sparse gradients in backprop)

### **26.5 Thrust - High-Level Algorithms**
- Parallel STL-like Algorithms
- Device Vectors
- Transformations and Reductions
- Sorting and Searching

📄 *Example Code:* `thrust_matmul.cu` (matrix operations with Thrust), `thrust_backprop.cu` (mini-batch processing)

---

## 🔄 Part 27: Multi-GPU Programming

**Goal**: Scale applications across multiple GPUs.

- **27.1** Multi-GPU Architecture and Topology
- **27.2** Device Management and Context
- **27.3** Peer-to-Peer Communication
- **27.4** Unified Virtual Addressing (UVA)
- **27.5** NCCL for Collective Operations
- **27.6** Multi-Process Service (MPS)
- **27.7** Load Balancing Strategies

📄 *Example Code:* `matmul_multigpu.cu` (distributed matrix multiplication), `backprop_data_parallel.cu` (data parallel training)

---

## 👥 Part 28: Cooperative Groups Advanced

**Goal**: Advanced thread cooperation patterns and optimizations.

- **28.1** Multi-Grid Synchronization
- **28.2** Dynamic Group Formation
- **28.3** Warp-Level Collectives
- **28.4** Complex Reduction Patterns
- **28.5** Producer-Consumer Patterns
- **28.6** Thread Block Clusters (SM 9.0+)

📄 *Example Code:* `multi_grid.cu`, `dynamic_groups.cu`, `warp_collectives.cu`, `cluster_communication.cu`

---

## 🤖 Part 29: Introduction to cuDNN (Optional)

**Goal**: Deep learning primitives with cuDNN (if working with neural networks).

- **29.1** cuDNN Overview and Setup
- **29.2** Convolution Operations
- **29.3** Pooling and Activation Functions
- **29.4** Batch Normalization
- **29.5** Performance Auto-tuning

📄 *Example Code:* `cudnn_layer.cu` (complete layer with forward/backward), `cudnn_cnn.cu` (simple CNN implementation)

---

## ✅ Summary

This intermediate section covers:

1. **Advanced Memory**: Unified memory, pinned memory, zero-copy techniques
2. **Parallelism**: Streams, async execution, synchronization
3. **I/O Optimization**: GPUDirect technologies for storage and networking
4. **CUDA Libraries**: cuBLAS, cuFFT, cuRAND, cuSPARSE, Thrust
5. **Multi-GPU**: Scaling across multiple devices
6. **Performance**: Advanced optimization techniques

**Prerequisites**: Complete [10.cuda_basic](../10.cuda_basic/README.md) (Parts 1-9) first.

**Next Steps**: For cutting-edge topics like CUTLASS, Triton, and custom kernels for transformers, see advanced tutorials.

---

## 📚 Resources

### NVIDIA Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Streams and Concurrency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)

### CUDA Libraries
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [cuRAND Documentation](https://docs.nvidia.com/cuda/curand/)
- [cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [Thrust Documentation](https://docs.nvidia.com/cuda/thrust/)
- [CUB Documentation](https://nvlabs.github.io/cub/)

### Advanced Topics
- [GPUDirect Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [cuDNN Documentation](https://docs.nvidia.com/cudnn/)

---