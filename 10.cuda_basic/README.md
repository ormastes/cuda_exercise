# 🚀 Basic CUDA Programming Tutorial (Code-Based)

> **Note**: This section covers fundamental CUDA concepts and basic programming techniques. Each chapter (except Part 1) introduces concepts via working example code.

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

📄 *Example Code:* `error_handling_demo.cu`

---

## 🧠 Part 7: Memory Hierarchy

**Goal**: Understanding CUDA memory types, bandwidth, latency, and basic optimization strategies.

- **7.1** Global Memory Characteristics
- **7.2** Shared Memory Basics
- **7.3** Constant and Texture Memory
- **7.4** Register Usage
- **7.5** Memory Coalescing Introduction
- **7.6** Basic Bank Conflicts

📄 *Example Code:* `matrix_multiply.cu` (evolves from vector_add to matrix multiplication, demonstrates coalescing)

---

## 🧵 Part 8: Thread Hierarchy in Practice

**Goal**: Demonstrate thread, warp, block, SM and grid concepts with practical examples.

- **8.1** `blockDim`, `threadIdx`, `gridDim`, `blockIdx`
- **8.2** 1D / 2D / 3D Kernels
- **8.3** Thread Indexing in Image Processing
- **8.4** Warp Basics and Divergence

📄 *Example Code:* `matrix_multiply_tiled.cu` (adds shared memory tiling to matrix multiplication)

---

## 🗃️ Part 9: CUDA Memory API

**Goal**: Master basic CUDA memory management APIs.

- **9.1** `cudaMalloc`, `cudaFree`
- **9.2** `cudaMemcpy` and Transfer Directions
- **9.3** `cudaMemset` and Initialization
- **9.4** Pageable vs Pinned Memory Basics
- **9.5** Error Handling in Memory Operations

📄 *Example Code:* `matrix_multiply_pinned.cu` (demonstrates pinned memory for faster transfers)

---

## ✅ Summary

This section covered the fundamental concepts of CUDA programming:

1. **Foundations**: Understanding GPU architecture and CUDA programming model
2. **First Kernel**: Writing and launching basic CUDA kernels
3. **Debugging**: Using cuda-gdb and VSCode for debugging
4. **Profiling**: Performance analysis with Nsight tools
5. **Testing**: Unit testing CUDA code with Google Test
6. **Error Handling**: Robust error checking and recovery strategies
7. **Memory Hierarchy**: Understanding different memory types
8. **Thread Hierarchy**: Working with threads, blocks, and grids
9. **Memory API**: Basic memory management operations

**Next Steps**: Continue with [20.intermediate_cuda](../20.intermediate_cuda/README.md) for advanced topics including streams, unified memory, and CUDA libraries.

---
