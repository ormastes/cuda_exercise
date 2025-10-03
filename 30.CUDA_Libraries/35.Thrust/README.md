# Thrust - High-Level Algorithms

**Goal**: Use C++ STL-like parallel algorithms for rapid CUDA development.

## Topics Covered

- Parallel STL-like Algorithms
- Device Vectors
- Transformations and Reductions
- Sorting and Searching

## Example Code

- `thrust_matmul.cu` - Matrix operations with Thrust
- `thrust_backprop.cu` - Mini-batch processing

## Key Concepts

### Container Types
- `thrust::device_vector` - GPU memory container
- `thrust::host_vector` - CPU memory container
- Automatic memory management

### Algorithms
- Transformations: `transform`, `for_each`
- Reductions: `reduce`, `transform_reduce`
- Prefix sums: `inclusive_scan`, `exclusive_scan`
- Sorting: `sort`, `sort_by_key`
- Searching: `find`, `binary_search`

### Execution Policies
- Sequential execution
- Parallel execution on device
- Custom execution policies

### Performance Tips
- Use raw pointers for kernel interop
- Leverage fusion for complex operations
- Consider CUB for lower-level control