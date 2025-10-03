# cuFFT - Fast Fourier Transforms

**Goal**: Leverage NVIDIA's optimized FFT library for signal processing and scientific computing.

## Topics Covered

- 1D, 2D, and 3D Transforms
- Real and Complex Transforms
- Batched FFTs
- Performance Optimization

## Example Code

- `cufft_convolution.cu` - FFT-based convolution
- `cufft_image_filter.cu` - FFT-based image processing

## Key Concepts

### Transform Types
- Complex-to-Complex (C2C)
- Real-to-Complex (R2C)
- Complex-to-Real (C2R)

### Plan Creation and Execution
- Creating FFT plans
- Executing forward and inverse transforms
- Managing workspace memory

### Performance Considerations
- Optimal FFT sizes (powers of 2)
- Batched transforms for multiple signals
- In-place vs out-of-place transforms