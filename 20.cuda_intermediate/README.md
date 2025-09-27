# 🚀 Intermediate CUDA Programming Tutorial

> **Note**: This section covers intermediate and advanced CUDA topics including advanced memory management, streams, and NVIDIA libraries.
>
> **Prerequisites**: Complete [10.cuda_basic](../10.cuda_basic/README.md) (Parts 1-9) first.

---

## 🔒 Part 10: Synchronization and Atomics

**Goal**: Advanced thread synchronization, atomic operations, and lock-free algorithms.

- **10.1** Thread Synchronization Primitives
- **10.2** Atomic Operations and Memory Ordering
- **10.3** Lock-Free Data Structures
- **10.4** Barriers and Memory Fences
- **10.5** CUB Library for Reductions
- **10.6** Avoiding Deadlocks and Race Conditions

📄 *Example Code:* `matrix_multiply_atomic.cu` (uses atomics for accumulation), `parallel_reduction.cu`

---

## 🧮 Part 11: Streams and Asynchronous Execution

**Goal**: Master CUDA streams for overlapping computation and data transfer.

- **11.1** Understanding CUDA Streams
- **11.2** Creating and Managing Streams
- **11.3** Async Memory Copy and Kernel Execution
- **11.4** Stream Priorities
- **11.5** Events for Synchronization and Timing
- **11.6** Multi-Stream Patterns
- **11.7** CUDA Graph API

📄 *Example Code:* `matrix_multiply_streams.cu` (parallel matrix operations with streams), `pipeline_compute.cu`

---

## 📊 Part 12: Unified Memory

**Goal**: Simplify memory management with Unified Memory and understand performance implications.

- **12.1** Unified Memory Architecture
- **12.2** `cudaMallocManaged` and Access Patterns
- **12.3** Memory Migration and Page Faults
- **12.4** Prefetching with `cudaMemPrefetchAsync`
- **12.5** Memory Advise with `cudaMemAdvise`
- **12.6** Performance Optimization Strategies
- **12.7** Multi-GPU with Unified Memory

📄 *Example Code:* `matrix_multiply_unified.cu` (unified memory version), `um_batch_process.cu`

---

## 📌 Part 13: Pinned Memory and Zero-Copy

**Goal**: Optimize host-device communication with advanced memory techniques.

- **13.1** Pinned Memory Deep Dive
- **13.2** `cudaHostAlloc` and Flags
- **13.3** Write-Combining Memory
- **13.4** Mapped Memory (Zero-Copy)
- **13.5** Portable Pinned Memory
- **13.6** Host Memory Registration
- **13.7** Performance Benchmarking

📄 *Example Code:* `matrix_multiply_zerocopy.cu` (zero-copy memory access), `pinned_transfer_bench.cu`

---

## 🖴 Part 14: GPUDirect and Storage (cuFile & GDS)

**Goal**: High-performance I/O with GPUDirect technologies.

- **14.1** GPUDirect Technology Overview
- **14.2** GPUDirect Storage (GDS) Architecture
- **14.3** cuFile API Introduction
- **14.4** Direct NVMe to GPU Transfers
- **14.5** GPUDirect RDMA for Network
- **14.6** GPUDirect P2P for Multi-GPU
- **14.7** Performance Optimization and Benchmarking

📄 *Example Code:* `matrix_multiply_multigpu.cu` (multi-GPU with P2P), `cufile_batch_load.cu`

---

## 🧩 Part 15: Using CUDA Libraries

**Goal**: Leverage NVIDIA's optimized libraries for common operations.

### **15.1 cuBLAS - Linear Algebra**
- Basic Operations (Level 1, 2, 3 BLAS)
- Matrix Multiplication with `cublasSgemm`
- Batched Operations
- Mixed Precision with Tensor Cores

📄 *Example Code:* `matrix_multiply_cublas.cu` (comparing custom kernel vs cuBLAS), `backprop_layer.cu` (neural network layer forward/backward)

### **15.2 cuFFT - Fast Fourier Transforms**
- 1D, 2D, and 3D Transforms
- Real and Complex Transforms
- Batched FFTs
- Performance Optimization

📄 *Example Code:* `cufft_convolution.cu`, `cufft_image_filter.cu` (FFT-based image processing)

### **15.3 cuRAND - Random Number Generation**
- Pseudo-Random Generators
- Quasi-Random Generators
- Distribution Functions
- Monte Carlo Simulations

📄 *Example Code:* `backprop_init.cu` (weight initialization for neural networks), `curand_monte_carlo.cu`

### **15.4 cuSPARSE - Sparse Matrix Operations**
- Sparse Matrix Formats (CSR, COO)
- Sparse Matrix-Vector Multiplication
- Format Conversions

📄 *Example Code:* `sparse_matmul.cu` (sparse matrix multiplication), `sparse_gradient.cu` (sparse gradients in backprop)

### **15.5 Thrust - High-Level Algorithms**
- Parallel STL-like Algorithms
- Device Vectors
- Transformations and Reductions
- Sorting and Searching

📄 *Example Code:* `thrust_matmul.cu` (matrix operations with Thrust), `thrust_backprop.cu` (mini-batch processing)

---

## 🔄 Part 16: Multi-GPU Programming

**Goal**: Scale applications across multiple GPUs.

- **16.1** Multi-GPU Architecture and Topology
- **16.2** Device Management and Context
- **16.3** Peer-to-Peer Communication
- **16.4** Unified Virtual Addressing (UVA)
- **16.5** NCCL for Collective Operations
- **16.6** Multi-Process Service (MPS)
- **16.7** Load Balancing Strategies

📄 *Example Code:* `matmul_multigpu.cu` (distributed matrix multiplication), `backprop_data_parallel.cu` (data parallel training)

---

## ⚡ Part 17: Performance Optimization Techniques

**Goal**: Advanced optimization strategies for production code.

- **17.1** Kernel Fusion and Loop Unrolling
- **17.2** Instruction-Level Parallelism (ILP)
- **17.3** Tensor Core Programming
- **17.4** Warp Shuffle Operations
- **17.5** Dynamic Parallelism
- **17.6** Occupancy Optimization
- **17.7** Optimization Case Studies

📄 *Example Code:* `matmul_tensor_core.cu` (tensor core optimized matmul), `backprop_fused.cu` (fused forward-backward kernels)

---

## 🤖 Part 18: Introduction to cuDNN (Optional)

**Goal**: Deep learning primitives with cuDNN (if working with neural networks).

- **18.1** cuDNN Overview and Setup
- **18.2** Convolution Operations
- **18.3** Pooling and Activation Functions
- **18.4** Batch Normalization
- **18.5** Performance Auto-tuning

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