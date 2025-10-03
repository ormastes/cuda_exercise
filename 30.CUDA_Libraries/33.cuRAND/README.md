# cuRAND - Random Number Generation

**Goal**: Generate high-quality random numbers on the GPU for simulations and machine learning.

## Topics Covered

- Pseudo-Random Generators
- Quasi-Random Generators
- Distribution Functions
- Monte Carlo Simulations

## Example Code

- `backprop_init.cu` - Weight initialization for neural networks
- `curand_monte_carlo.cu` - Monte Carlo simulation examples

## Key Concepts

### Generator Types
- XORWOW (default, fast)
- MRG32k3a (good for parallel streams)
- MTGP32 (Mersenne Twister)
- Sobol (quasi-random)

### Distributions
- Uniform distribution
- Normal (Gaussian) distribution
- Log-normal distribution
- Poisson distribution

### Usage Patterns
- Host API for bulk generation
- Device API for in-kernel generation
- Seed management for reproducibility