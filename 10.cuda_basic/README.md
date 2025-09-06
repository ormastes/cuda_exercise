# 🚀 Basic CUDA Programming Tutorial (Code-Based)

> **Note**: Each chapter (except Part 1) introduces concepts or APIs via working example code.

---

## 🧩 Part 1: Foundations *(No Code)*

Build a conceptual base before writing code. See [11.Foundations/README.md](11.Foundations/README.md) for detailed content.

- **1.1** [What is CUDA?](11.Foundations/README.md#11-what-is-cuda)
- **1.2** [CUDA vs CPU programming](11.Foundations/README.md#12-cuda-vs-cpu-programming)
- **1.3** [Warp as Shared Environment](11.Foundations/README.md#13-warp-as-shared-environment)
- **1.4** [CUDA Architecture](11.Foundations/README.md#14-cuda-architecture)  
  (threads, warps, blocks, grids)
- **1.5** [Memory Hierarchy](11.Foundations/README.md#15-memory-hierarchy)  
  (global, shared, local, texture, constant)
- **1.6** [CUDA Toolchain](11.Foundations/README.md#16-cuda-toolchain-overview)  
  (`nvcc`, runtime API, driver API)
- **1.7** [Hardware Requirements & Setup](11.Foundations/README.md#17-hardware-requirements--setup)
- **1.8** [NVIDIA Driver and CUDA Installation on Ubuntu 24.04](11.Foundations/README.md#18-nvidia-driver-and-cuda-installation-on-ubuntu-2404)
- **1.9** [Installing Clang 20 on Ubuntu 24.04](11.Foundations/README.md#19-installing-clang-20-on-ubuntu-2404)
- **1.10** [Installing Ubuntu on WSL2 (Windows Users)](11.Foundations/README.md#110-installing-ubuntu-on-wsl2-windows-users)

---

## ⚙️ Part 2: Your First CUDA Kernel

**Goal**: Introduce kernel syntax, compilation, and launch.

- **2.1** Host vs Device Code Separation
- **2.2** `__global__` Function
- **2.3** Launch Configuration
- **2.4** Basic Vector Addition

📄 *Example Code:* `vector_add.cu`

---

## 🧵 Part 3: Thread Hierarchy in Practice

**Goal**: Demonstrate how threads, blocks, and grids affect execution.

- **3.1** `blockDim`, `threadIdx`, `gridDim`, `blockIdx`
- **3.2** 1D / 2D / 3D Kernels
- **3.3** Thread Indexing in Image Processing

📄 *Example Code:* `image_negation_2D.cu`

---

## 🧠 Part 4: Shared Memory & Synchronization

**Goal**: Leverage shared memory for performance.

- **4.1** Shared Memory Usage
- **4.2** Bank Conflicts & Optimizations
- **4.3** `__syncthreads()` and Race Conditions

📄 *Example Code:* `matrix_mul_shared.cu`

---

## 🗃️ Part 5: CUDA Memory API

**Goal**: Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`.

- **5.1** Allocating Memory on Device
- **5.2** Copying Memory Between Host & Device
- **5.3** Pageable vs Pinned Memory

📄 *Example Code:* `cuda_memory_api_demo.cu`

---

## 🧮 Part 6: Streams and Asynchronous Execution

**Goal**: Introduce `cudaStream_t` and overlapped copy/compute.

- **6.1** Using Streams to Avoid Serialization
- **6.2** Async Memory Copy
- **6.3** Measuring Performance with Events

📄 *Example Code:* `stream_overlap_example.cu`

---

## 🧪 Part 7: Error Handling and Debugging

**Goal**: Improve reliability of CUDA apps.

- **7.1** `cudaError_t` and Error Check Macros
- **7.2** Debugging Tools: `cuda-gdb`, Nsight
- **7.3** Detecting Race Conditions

📄 *Example Code:*  
- `error_checking_macros.h`  
- `race_condition_demo.cu`

---

## 📊 Part 8: Unified Memory and `cudaMallocManaged`

**Goal**: Introduce Unified Memory for simplified data access.

- **8.1** Pros and Cons of Unified Memory
- **8.2** Using `cudaMallocManaged`
- **8.3** Memory Migration and Prefetching

📄 *Example Code:* `unified_memory_vector_add.cu`

---

## 🖴 Part 9: CUDA and NVMe (cuFile & GDS Intro)

**Goal**: Bridge to high-performance I/O with GPU & storage.

- **9.1** Intro to GPUDirect Storage
- **9.2** Overview of `cuFile` API
- **9.3** Example: Reading Data from SSD to GPU Memory

📄 *Example Code:* `cufile_read_example.cu`

---

## 🧩 Part 10: Using CUDA Libraries (cuBLAS, cuFFT, etc.)

**Goal**: Leverage NVIDIA's optimized libraries.

- **10.1** cuBLAS: Matrix Multiplication
- **10.2** cuFFT: Signal Transformation
- **10.3** cuRAND: Random Number Generation

📄 *Example Code:*  
- `cublas_gemm.cu`  
- `cufft_demo.cu`  
- `curand_demo.cu`

---
