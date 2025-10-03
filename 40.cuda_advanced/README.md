# Part 30: Advanced CUDA Programming

Cutting-edge CUDA features and specialized optimization techniques for modern GPU architectures.

## Topics

### [31. Advanced PTX Assembly](31.Advanced_PTX_Assembly/)
- Inline PTX programming
- Low-level GPU control
- Hardware-specific optimizations

### [32. Compiler Optimizations](32.Compiler_Optimizations/)
- NVCC optimization flags
- Register management
- Instruction-level parallelism

### [33. CUDA Intrinsics](33.CUDA_Intrinsics/)
- Warp-level primitives
- Fast math functions
- Hardware intrinsics

### [34. CUDA Graphs](34.CUDA_Graphs/)
- Graph creation and execution
- Stream capture
- Kernel launch optimization

### [35. Tensor Cores](35.Tensor_Cores/)
- WMMA API
- Mixed-precision computing
- AI/ML acceleration

### [36. CUDA IPC](36.CUDA_IPC/)
- Inter-process communication
- Shared memory handles
- Multi-process GPU apps

### [37. Virtual Memory](37.Virtual_Memory/)
- Virtual address management
- Memory mapping
- Sparse allocation

### [38. Hardware Scheduling](38.Hardware_Scheduling/)
- GPU scheduler behavior
- Occupancy optimization
- Load balancing strategies

### [39. Tile-Based Programming](39.Tile_Based_Programming/)
- Tile-based programming model (CUDA 13.0+)
- Complementary to thread-based model
- Array-level operations

### [40. Zstd Compression](40.Zstd_Compression/)
- Binary compression (CUDA 12.0+)
- Reduced deployment sizes
- Runtime decompression

## Prerequisites

- Complete understanding of Parts 10 (Basic) and 20 (Intermediate)
- CUDA Toolkit 12.0 or higher (13.0+ for some features)
- GPU with Compute Capability 7.0+ (some features require 8.0+ or 9.0+)
- Nsight Systems and Nsight Compute for profiling

## Building

Each subdirectory contains example code and can be built with:

```bash
cd build
cmake ..
ninja

# Run examples
cd 30.cuda_advanced/31.Advanced_PTX_Assembly
./31_PTXAssembly_inline_ptx
```

## Learning Path

1. Start with PTX Assembly and Compiler Optimizations for low-level understanding
2. Master CUDA Intrinsics for hardware-level control
3. Learn CUDA Graphs for kernel launch optimization
4. Explore Tensor Cores for AI/ML workloads
5. Study advanced Multi-GPU patterns for scalable applications
6. Explore modern features like Tile-Based Programming (CUDA 13.0+)

## Key Concepts

- **Low-Level Control**: PTX assembly and compiler optimizations
- **Hardware Features**: Direct access to modern GPU capabilities
- **Launch Optimization**: Advanced techniques with CUDA Graphs
- **Modern Architectures**: Features for SM 7.0+ and 8.0+ GPUs
- **Specialized Computing**: Tensor Cores, IPC, Virtual Memory
- **New Programming Models**: Tile-based complementing thread-based

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Tools Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)