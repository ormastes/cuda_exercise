# cuBLAS - Linear Algebra

**Goal**: Master NVIDIA's optimized BLAS library for linear algebra operations on GPUs.

## Topics Covered

- Basic Operations (Level 1, 2, 3 BLAS)
- Matrix Multiplication with `cublasSgemm`
- Batched Operations
- Mixed Precision with Tensor Cores

## Example Code

- `matrix_multiply_cublas.cu` - Comparing custom kernel vs cuBLAS
- `backprop_layer.cu` - Neural network layer forward/backward

## Key Concepts

### Level 1 BLAS Operations
- Vector operations (dot product, norm, scaling)

### Level 2 BLAS Operations
- Matrix-vector operations

### Level 3 BLAS Operations
- Matrix-matrix operations (GEMM)

### Performance Tips
- Use column-major order for best performance
- Leverage Tensor Cores with mixed precision
- Use batched operations for multiple small matrices