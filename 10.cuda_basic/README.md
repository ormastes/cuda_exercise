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

**Goal**: Introduce kernel syntax, compilation, launch, and VSCode setup. See [12.Your First CUDA Kernel/README.md](12.Your%20First%20CUDA%20Kernel/README.md) for detailed content.

- **2.1** [Host vs Device Code Separation](12.Your%20First%20CUDA%20Kernel/README.md#21-host-vs-device-code-separation)
- **2.2** [`__global__`, `__device__`, and `dim3` API](12.Your%20First%20CUDA%20Kernel/README.md#22-__global__-__device__-and-dim3-api)
- **2.3** [Launch Configuration with `dim3`](12.Your%20First%20CUDA%20Kernel/README.md#23-launch-configuration-with-dim3)
- **2.4** [CUDA Memory Management APIs](12.Your%20First%20CUDA%20Kernel/README.md#24-cuda-memory-management-apis)
- **2.5** [Vector Add Example](12.Your%20First%20CUDA%20Kernel/README.md#25-vector-add-example)
- **2.6** [VSCode Preset Selection & Build Setup](12.Your%20First%20CUDA%20Kernel/README.md#26-vscode-preset-selection--build-setup)
- **2.7** [Configuration Files Explained](12.Your%20First%20CUDA%20Kernel/README.md#27-configuration-files-explained)
- **2.8** [API Recap](12.Your%20First%20CUDA%20Kernel/README.md#28-api-recap)
- **2.9** [Summary](12.Your%20First%20CUDA%20Kernel/README.md#29-summary)

📄 *Example Code:* `vector_add_2d.cu`

---

## 🐞 Part 3: Debugging CUDA in VSCode

**Goal**: Master GPU debugging techniques using NVIDIA Nsight and cuda-gdb in VSCode. See [13.Debugging CUDA in VSCode/README.md](13.Debugging%20CUDA%20in%20VSCode/README.md) for detailed content.

- **3.1** [Overview of CUDA Debugging](13.Debugging%20CUDA%20in%20VSCode/README.md#31-overview-of-cuda-debugging)
- **3.2** [VSCode Debug Configuration](13.Debugging%20CUDA%20in%20VSCode/README.md#32-vscode-debug-configuration)
- **3.3** [Debugging Workflow](13.Debugging%20CUDA%20in%20VSCode/README.md#33-debugging-workflow)
- **3.4** [Debug Features and Commands](13.Debugging%20CUDA%20in%20VSCode/README.md#34-debug-features-and-commands)
- **3.5** [Thread and Block Navigation](13.Debugging%20CUDA%20in%20VSCode/README.md#35-thread-and-block-navigation)
- **3.6** [Common Debugging Scenarios](13.Debugging%20CUDA%20in%20VSCode/README.md#36-common-debugging-scenarios)
- **3.7** [Debug Output and Logging](13.Debugging%20CUDA%20in%20VSCode/README.md#37-debug-output-and-logging)
- **3.8** [Performance Debugging](13.Debugging%20CUDA%20in%20VSCode/README.md#38-performance-debugging)
- **3.9** [Troubleshooting Common Issues](13.Debugging%20CUDA%20in%20VSCode/README.md#39-troubleshooting-common-issues)
- **3.10** [Advanced Debugging Tips](13.Debugging%20CUDA%20in%20VSCode/README.md#310-advanced-debugging-tips)
- **3.11** [Debug Commands Reference](13.Debugging%20CUDA%20in%20VSCode/README.md#311-debug-commands-reference)
- **3.12** [Summary](13.Debugging%20CUDA%20in%20VSCode/README.md#312-summary)

---

## 🔍 Part 4: Code Inspection and Profiling

**Goal**: Analyze and optimize CUDA code performance using NVIDIA profiling tools. See [14.Code Inspection and Profiling/README.md](14.Code%20Inspection%20and%20Profiling/README.md) for detailed content.

- **4.1** [Introduction to CUDA Profiling](14.Code%20Inspection%20and%20Profiling/README.md#41-introduction-to-cuda-profiling)
- **4.2** [Using Nsight Compute](14.Code%20Inspection%20and%20Profiling/README.md#42-using-nsight-compute)
- **4.3** [Using Nsight Systems](14.Code%20Inspection%20and%20Profiling/README.md#43-using-nsight-systems)
- **4.4** [Kernel Performance Metrics](14.Code%20Inspection%20and%20Profiling/README.md#44-kernel-performance-metrics)
- **4.5** [Memory Access Patterns](14.Code%20Inspection%20and%20Profiling/README.md#45-memory-access-patterns)
- **4.6** [Occupancy Analysis](14.Code%20Inspection%20and%20Profiling/README.md#46-occupancy-analysis)
- **4.7** [Identifying Bottlenecks](14.Code%20Inspection%20and%20Profiling/README.md#47-identifying-bottlenecks)
- **4.8** [Optimization Strategies](14.Code%20Inspection%20and%20Profiling/README.md#48-optimization-strategies)
- **4.9** [Profiling Best Practices](14.Code%20Inspection%20and%20Profiling/README.md#49-profiling-best-practices)
- **4.10** [Summary](14.Code%20Inspection%20and%20Profiling/README.md#410-summary)

📄 *Example Code:* Various optimization examples

---

## 🧪 Part 5: Unit Testing CUDA Code

**Goal**: Implement comprehensive testing for CUDA kernels using Google Test framework. See [15.Unit Testing/README.md](15.Unit%20Testing/README.md) for detailed content.

- **5.1** [Introduction to GPU Testing](15.Unit%20Testing/README.md#51-introduction-to-gpu-testing)
- **5.2** [Google Test Integration](15.Unit%20Testing/README.md#52-google-test-integration)
- **5.3** [Basic GPU Test Macros](15.Unit%20Testing/README.md#53-basic-gpu-test-macros)
- **5.4** [Parameterized GPU Tests](15.Unit%20Testing/README.md#54-parameterized-gpu-tests)
- **5.5** [Generator-Based Testing](15.Unit%20Testing/README.md#55-generator-based-testing)
- **5.6** [Testing Best Practices](15.Unit%20Testing/README.md#56-testing-best-practices)
- **5.7** [Debugging Failed Tests](15.Unit%20Testing/README.md#57-debugging-failed-tests)
- **5.8** [Performance Testing](15.Unit%20Testing/README.md#58-performance-testing)
- **5.9** [Test Coverage Analysis](15.Unit%20Testing/README.md#59-test-coverage-analysis)
- **5.10** [Summary](15.Unit%20Testing/README.md#510-summary)

📄 *Example Code:* `device_test.cu`, `host_test.cu`, test samples

---

## 🛡️ Part 6: Error Handling and Debugging

**Goal**: Build robust CUDA applications with comprehensive error handling. See [16.Error Handling and Debugging/README.md](16.Error%20Handling%20and%20Debugging/README.md) for detailed content.

- **6.1** [CUDA Error Types and Codes](16.Error%20Handling%20and%20Debugging/README.md#61-cuda-error-types-and-codes)
- **6.2** [Error Checking Macros](16.Error%20Handling%20and%20Debugging/README.md#62-error-checking-macros)
- **6.3** [Synchronous vs Asynchronous Errors](16.Error%20Handling%20and%20Debugging/README.md#63-synchronous-vs-asynchronous-errors)
- **6.4** [Debugging Memory Errors](16.Error%20Handling%20and%20Debugging/README.md#64-debugging-memory-errors)
- **6.5** [Race Condition Detection](16.Error%20Handling%20and%20Debugging/README.md#65-race-condition-detection)
- **6.6** [Deadlock Prevention](16.Error%20Handling%20and%20Debugging/README.md#66-deadlock-prevention)
- **6.7** [Error Recovery Strategies](16.Error%20Handling%20and%20Debugging/README.md#67-error-recovery-strategies)
- **6.8** [Logging and Diagnostics](16.Error%20Handling%20and%20Debugging/README.md#68-logging-and-diagnostics)
- **6.9** [Production Error Handling](16.Error%20Handling%20and%20Debugging/README.md#69-production-error-handling)
- **6.10** [Summary](16.Error%20Handling%20and%20Debugging/README.md#610-summary)

📄 *Example Code:* Error handling utilities and examples

---

# 🚀 Planned Future Content

> **Note**: The following sections are planned for future development and will be added as the tutorial expands.

---

## 🧠 Part 7: Memory Hierarchy

what memory bandwidth, size, latency, usage. bank conflicts, coalescing, etc.

---

## 🧵 Part 8: Thread Hierarchy in Practice

**Goal**: Demonstrate thread, warp, block, SM and grid concepts with 2D image processing.

- **8.1** `blockDim`, `threadIdx`, `gridDim`, `blockIdx`
- **8.2** 1D / 2D / 3D Kernels
- **8.3** Thread Indexing in Image Processing

📄 *Example Code:* `image_negation_2D.cu`

---

## 🗃️ Part 9: CUDA Basic Memory API

**Goal**: Use `cudaMalloc`, `cudaMemcpy`, `cudaFree`.

- **9.1** Allocating Memory on Device
- **9.2** Copying Memory Between Host & Device
- **9.3** Pageable vs Pinned Memory

📄 *Example Code:* `cuda_memory_api_demo.cu`

---


## 🧠 Part 10: Memory Lock, async and Synchronization

all cuda lock, async with CUB and synchronization aspects. 


## 🧮 Part 11: Streams and Asynchronous Execution

**Goal**: Introduce `cudaStream_t` and overlapped copy/compute.

- **11.1** Using Streams to Avoid Serialization
- **11.2** Async Memory Copy
- **11.3** Measuring Performance with Events

📄 *Example Code:* `stream_overlap_example.cu`

---



## 📊 Part 12: Unified Memory and `cudaMallocManaged`

**Goal**: Introduce Unified Memory for simplified data access.

- **12.1** Pros and Cons of Unified Memory
- **12.2** Using `cudaMallocManaged`
- **12.3** Memory Migration and Prefetching

📄 *Example Code:* `unified_memory_vector_add.cu`

---

## 📊 Part 13: Pin host/CUDA memory



---

## 🖴 Part 14: CUDA and NVMe (cuFile & GDS Intro)

**Goal**: Bridge to high-performance I/O with GPU & storage.

- **14.1** Intro to GPUDirect Storage
- **14.2** Overview of `cuFile` API
- **14.3** Example: Reading Data from SSD to GPU Memory

📄 *Example Code:* `cufile_read_example.cu`

---

## 🧩 Part 15: Using CUDA Libraries (cuBLAS, cuFFT, etc.)

**Goal**: Leverage NVIDIA's optimized libraries.

- **15.1** cuBLAS: Matrix Multiplication
- **15.2** cuFFT: Signal Transformation
- **15.3** cuRAND: Random Number Generation

📄 *Example Code:*  
- `cublas_gemm.cu`  
- `cufft_demo.cu`  
- `curand_demo.cu`

---
