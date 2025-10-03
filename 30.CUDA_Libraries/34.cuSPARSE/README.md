# cuSPARSE - Sparse Matrix Operations

**Goal**: Efficiently work with sparse matrices using NVIDIA's optimized sparse linear algebra library.

## Topics Covered

- Sparse Matrix Formats (CSR, COO)
- Sparse Matrix-Vector Multiplication
- Format Conversions

## Example Code

- `sparse_matmul.cu` - Sparse matrix multiplication
- `sparse_gradient.cu` - Sparse gradients in backpropagation

## Key Concepts

### Sparse Matrix Formats
- COO (Coordinate format)
- CSR (Compressed Sparse Row)
- CSC (Compressed Sparse Column)
- BSR (Block Sparse Row)

### Core Operations
- SpMV (Sparse Matrix-Vector multiplication)
- SpMM (Sparse Matrix-Matrix multiplication)
- Sparse triangular solve
- Format conversions

### Performance Tips
- Choose appropriate format for access pattern
- Use structured sparsity when possible
- Consider hybrid formats for best performance