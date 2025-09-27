# 🚀 Basic CUDA Programming Tutorial (Code-Based)

> **Note**: This section covers fundamental CUDA concepts and basic programming techniques. Each chapter (except Part 1) introduces concepts via working example code.

---

## 🧩 Part 11: Foundations *(No Code)*

Build a conceptual base before writing code. See [11.Foundations/README.md](11.Foundations/README.md) for detailed content.

- **11.1** [What is CUDA?](11.Foundations/README.md#11-what-is-cuda)
- **11.2** [CUDA vs CPU programming](11.Foundations/README.md#12-cuda-vs-cpu-programming)
- **11.3** [Warp as Shared Environment](11.Foundations/README.md#13-warp-as-shared-environment)
- **11.4** [CUDA Architecture](11.Foundations/README.md#14-cuda-architecture)  
  (threads, warps, blocks, grids)
- **11.5** [Memory Hierarchy](11.Foundations/README.md#15-memory-hierarchy)  
  (global, shared, local, texture, constant)
- **11.6** [CUDA Toolchain](11.Foundations/README.md#16-cuda-toolchain-overview)  
  (`nvcc`, runtime API, driver API)
- **11.7** [Hardware Requirements & Setup](11.Foundations/README.md#17-hardware-requirements--setup)
- **11.8** [NVIDIA Driver and CUDA Installation on Ubuntu 24.04](11.Foundations/README.md#18-nvidia-driver-and-cuda-installation-on-ubuntu-2404)
- **11.9** [Installing Clang 20 on Ubuntu 24.04](11.Foundations/README.md#19-installing-clang-20-on-ubuntu-2404)
- **11.10** [Installing Ubuntu on WSL2 (Windows Users)](11.Foundations/README.md#110-installing-ubuntu-on-wsl2-windows-users)

---

## ⚙️ Part 12: Your First CUDA Kernel

**Goal**: Introduce kernel syntax, compilation, launch, and VSCode setup. See [12.Your First CUDA Kernel/README.md](12.Your%20First%20CUDA%20Kernel/README.md) for detailed content.

- **12.1** [Host vs Device Code Separation](12.Your%20First%20CUDA%20Kernel/README.md#21-host-vs-device-code-separation)
- **12.2** [`__global__`, `__device__`, and `dim3` API](12.Your%20First%20CUDA%20Kernel/README.md#22-__global__-__device__-and-dim3-api)
- **12.3** [Launch Configuration with `dim3`](12.Your%20First%20CUDA%20Kernel/README.md#23-launch-configuration-with-dim3)
- **12.4** [CUDA Memory Management APIs](12.Your%20First%20CUDA%20Kernel/README.md#24-cuda-memory-management-apis)
- **12.5** [Vector Add Example](12.Your%20First%20CUDA%20Kernel/README.md#25-vector-add-example)
- **12.6** [VSCode Preset Selection & Build Setup](12.Your%20First%20CUDA%20Kernel/README.md#26-vscode-preset-selection--build-setup)
- **12.7** [Configuration Files Explained](12.Your%20First%20CUDA%20Kernel/README.md#27-configuration-files-explained)
- **12.8** [API Recap](12.Your%20First%20CUDA%20Kernel/README.md#28-api-recap)
- **12.9** [Summary](12.Your%20First%20CUDA%20Kernel/README.md#29-summary)

📄 *Example Code:* `vector_add_2d.cu`

---

## 🐞 Part 13: Debugging CUDA in VSCode

**Goal**: Master GPU debugging techniques using NVIDIA Nsight and cuda-gdb in VSCode. See [13.Debugging CUDA in VSCode/README.md](13.Debugging%20CUDA%20in%20VSCode/README.md) for detailed content.

- **13.1** [Overview of CUDA Debugging](13.Debugging%20CUDA%20in%20VSCode/README.md#31-overview-of-cuda-debugging)
- **13.2** [VSCode Debug Configuration](13.Debugging%20CUDA%20in%20VSCode/README.md#32-vscode-debug-configuration)
- **13.3** [Debugging Workflow](13.Debugging%20CUDA%20in%20VSCode/README.md#33-debugging-workflow)
- **13.4** [Debug Features and Commands](13.Debugging%20CUDA%20in%20VSCode/README.md#34-debug-features-and-commands)
- **13.5** [Thread and Block Navigation](13.Debugging%20CUDA%20in%20VSCode/README.md#35-thread-and-block-navigation)
- **13.6** [Common Debugging Scenarios](13.Debugging%20CUDA%20in%20VSCode/README.md#36-common-debugging-scenarios)
- **13.7** [Debug Output and Logging](13.Debugging%20CUDA%20in%20VSCode/README.md#37-debug-output-and-logging)
- **13.8** [Performance Debugging](13.Debugging%20CUDA%20in%20VSCode/README.md#38-performance-debugging)
- **13.9** [Troubleshooting Common Issues](13.Debugging%20CUDA%20in%20VSCode/README.md#39-troubleshooting-common-issues)
- **13.10** [Advanced Debugging Tips](13.Debugging%20CUDA%20in%20VSCode/README.md#310-advanced-debugging-tips)
- **13.11** [Debug Commands Reference](13.Debugging%20CUDA%20in%20VSCode/README.md#311-debug-commands-reference)
- **13.12** [Summary](13.Debugging%20CUDA%20in%20VSCode/README.md#312-summary)

---

## 🔍 Part 14: Code Inspection and Profiling

**Goal**: Analyze and optimize CUDA code performance using NVIDIA profiling tools. See [14.Code Inspection and Profiling/README.md](14.Code%20Inspection%20and%20Profiling/README.md) for detailed content.

- **14.1** [Introduction to CUDA Profiling](14.Code%20Inspection%20and%20Profiling/README.md#41-introduction-to-cuda-profiling)
- **14.2** [Using Nsight Compute](14.Code%20Inspection%20and%20Profiling/README.md#42-using-nsight-compute)
- **14.3** [Using Nsight Systems](14.Code%20Inspection%20and%20Profiling/README.md#43-using-nsight-systems)
- **14.4** [Kernel Performance Metrics](14.Code%20Inspection%20and%20Profiling/README.md#44-kernel-performance-metrics)
- **14.5** [Memory Access Patterns](14.Code%20Inspection%20and%20Profiling/README.md#45-memory-access-patterns)
- **14.6** [Occupancy Analysis](14.Code%20Inspection%20and%20Profiling/README.md#46-occupancy-analysis)
- **14.7** [Identifying Bottlenecks](14.Code%20Inspection%20and%20Profiling/README.md#47-identifying-bottlenecks)
- **14.8** [Optimization Strategies](14.Code%20Inspection%20and%20Profiling/README.md#48-optimization-strategies)
- **14.9** [Profiling Best Practices](14.Code%20Inspection%20and%20Profiling/README.md#49-profiling-best-practices)
- **14.10** [Summary](14.Code%20Inspection%20and%20Profiling/README.md#410-summary)

📄 *Example Code:* Various optimization examples

---

## 🧪 Part 15: Unit Testing CUDA Code

**Goal**: Implement comprehensive testing for CUDA kernels using Google Test framework. See [15.Unit Testing/README.md](15.Unit%20Testing/README.md) for detailed content.

- **15.1** [Introduction to GPU Testing](15.Unit%20Testing/README.md#51-introduction-to-gpu-testing)
- **15.2** [Google Test Integration](15.Unit%20Testing/README.md#52-google-test-integration)
- **15.3** [Basic GPU Test Macros](15.Unit%20Testing/README.md#53-basic-gpu-test-macros)
- **15.4** [Parameterized GPU Tests](15.Unit%20Testing/README.md#54-parameterized-gpu-tests)
- **15.5** [Generator-Based Testing](15.Unit%20Testing/README.md#55-generator-based-testing)
- **15.6** [Testing Best Practices](15.Unit%20Testing/README.md#56-testing-best-practices)
- **15.7** [Debugging Failed Tests](15.Unit%20Testing/README.md#57-debugging-failed-tests)
- **15.8** [Performance Testing](15.Unit%20Testing/README.md#58-performance-testing)
- **15.9** [Test Coverage Analysis](15.Unit%20Testing/README.md#59-test-coverage-analysis)
- **15.10** [Summary](15.Unit%20Testing/README.md#510-summary)

📄 *Example Code:* `device_test.cu`, `host_test.cu`, test samples

---

## 🛡️ Part 16: Error Handling and Debugging

**Goal**: Build robust CUDA applications with comprehensive error handling. See [16.Error Handling and Debugging/README.md](16.Error%20Handling%20and%20Debugging/README.md) for detailed content.

- **16.1** [CUDA Error Types and Codes](16.Error%20Handling%20and%20Debugging/README.md#61-cuda-error-types-and-codes)
- **16.2** [Error Checking Macros](16.Error%20Handling%20and%20Debugging/README.md#62-error-checking-macros)
- **16.3** [Synchronous vs Asynchronous Errors](16.Error%20Handling%20and%20Debugging/README.md#63-synchronous-vs-asynchronous-errors)
- **16.4** [Debugging Memory Errors](16.Error%20Handling%20and%20Debugging/README.md#64-debugging-memory-errors)
- **16.5** [Race Condition Detection](16.Error%20Handling%20and%20Debugging/README.md#65-race-condition-detection)
- **16.6** [Deadlock Prevention](16.Error%20Handling%20and%20Debugging/README.md#66-deadlock-prevention)
- **16.7** [Error Recovery Strategies](16.Error%20Handling%20and%20Debugging/README.md#67-error-recovery-strategies)
- **16.8** [Logging and Diagnostics](16.Error%20Handling%20and%20Debugging/README.md#68-logging-and-diagnostics)
- **16.9** [Production Error Handling](16.Error%20Handling%20and%20Debugging/README.md#69-production-error-handling)
- **16.10** [Summary](16.Error%20Handling%20and%20Debugging/README.md#610-summary)

📄 *Example Code:* `error_handling_demo.cu`

---

## 🧠 Part 17: Memory Hierarchy

**Goal**: Understanding CUDA memory types, bandwidth, latency, and basic optimization strategies.

- **17.1** Global Memory Characteristics
- **17.2** Shared Memory Basics
- **17.3** Constant and Texture Memory
- **17.4** Register Usage
- **17.5** Memory Coalescing Introduction
- **17.6** Basic Bank Conflicts

📄 *Example Code:* `matrix_multiply.cu` (evolves from vector_add to matrix multiplication, demonstrates coalescing)

---

## 🧵 Part 18: Thread Hierarchy in Practice

**Goal**: Demonstrate thread, warp, block, SM and grid concepts with practical examples.

- **18.1** `blockDim`, `threadIdx`, `gridDim`, `blockIdx`
- **18.2** 1D / 2D / 3D Kernels
- **18.3** Thread Indexing in Image Processing
- **18.4** Warp Basics and Divergence

📄 *Example Code:* `matrix_multiply_tiled.cu` (adds shared memory tiling to matrix multiplication)

---

## 🗃️ Part 19: CUDA Memory API

**Goal**: Master basic CUDA memory management APIs.

- **19.1** `cudaMalloc`, `cudaFree`
- **19.2** `cudaMemcpy` and Transfer Directions
- **19.3** `cudaMemset` and Initialization
- **19.4** Pageable vs Pinned Memory Basics
- **19.5** Error Handling in Memory Operations

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
