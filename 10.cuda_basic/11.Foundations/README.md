# 🧩 Part 11: Foundations

---

## 11.1 What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and API model.

It allows developers to:
- Offload computation to NVIDIA GPUs
- Use familiar C/C++/Fortran syntax with extensions
- Scale from small embedded devices to massive supercomputers

CUDA supports both:
- **CUDA C/C++**: Host-device style programming
- **CUDA Libraries**: cuBLAS, cuFFT, cuRAND, cuSPARSE, etc.

---

## 11.2 CUDA vs CPU Programming

| Feature        | CPU                      | CUDA GPU                        |
|----------------|---------------------------|----------------------------------|
| Core Count     | Few (2–64)                | Thousands (SIMT model)          |
| Parallelism    | Thread-level              | Warp/block/grid-level            |
| Memory Access  | Low latency, high cache   | High bandwidth, high latency     |
| Strength       | Control flow, branching   | Throughput, vector processing    |

In CUDA:
- Work is parallelized across **threads**
- Threads are grouped into **blocks**
- Blocks are grouped into a **grid**

---

## 11.3 Warp as Shared Environment

1. **Shared Instruction Pointer**: All 32 threads in a warp share the same program counter - they're literally executing the same instruction at the same time.

2. **Shared Execution Resources**:
   - One instruction fetch/decode unit per warp
   - Shared scheduling slot
   - Shared execution pipeline
   - When threads in a warp take different execution paths, the warp serially executes each branch path

3. **This is WHY**:
   - Threads in a warp move together through the program
   - Branch divergence hurts performance (the warp must serialize divergent paths)
   - Warp-level primitives work (`__ballot_sync()`, `__shfl_sync()`, etc.)

4. **Contrast with CPU threads**:
   - **CPU**: Each thread has independent execution context, own program counter
   - **GPU Warp**: 32 threads share execution context, move as one unit

**Practical Implication**:
```cuda
// These 32 threads (assuming block size >= 32) form one warp
// They share the execution environment
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;  // Position within the warp

// All threads in warp execute this together
if (condition) {
    // If even one thread takes this branch,
    // the whole warp must execute it
}
```

The warp IS the fundamental unit of execution - not individual threads. Threads are just lanes within the warp's shared execution environment.

---

## 11.4 CUDA Architecture

### Basic Building Blocks:
- **Thread**: Executes a single CUDA kernel instance
- **Warp**: Group of 32 threads executed in lockstep
- **Block**: Group of threads (1D/2D/3D)
- **Grid**: Group of blocks (1D/2D/3D)

```
Grid
└── Block (0)
└── Threads (0...n)
└── Block (1)
└── Threads (0...n)
```

### SIMT (Single Instruction, Multiple Threads)
Threads in a warp execute the same instruction. Divergence (e.g., `if` conditions) may cause slowdown.

---

## 11.5 Memory Hierarchy

| Type             | Scope              | Access Latency | Usage |
|------------------|--------------------|----------------|-------|
| **Global**       | All threads         | High           | Input/output data |
| **Shared**       | Block-local         | Low            | Fast cooperation between threads |
| **Local**        | Thread-local        | High           | Registers / local variables |
| **Constant**     | All threads (RO)    | Very Low       | Small constants |
| **Texture/Surface** | Special-purpose | Optimized      | Image data |

### Important Notes:
- Minimize global memory accesses
- Use shared memory for data reused across threads
- Register usage is limited per thread

---

## 11.6 CUDA Toolchain Overview

| Tool          | Purpose                          |
|---------------|----------------------------------|
| `nvcc`        | CUDA compiler (driver + runtime) |
| `cuda-gdb`    | CUDA debugger                    |
| Nsight        | Profiler, debugger, IDE          |
| `cuobjdump`   | Inspect binary for PTX/CUBIN     |

### CUDA APIs
- **Runtime API** (`cudaMalloc`, `cudaMemcpy`, etc.)
- **Driver API** (`cuMemAlloc`, `cuMemcpyHtoD`, etc.)

The runtime API is easier and covers most use cases.

---

## 11.7 Hardware Requirements & Setup

### Minimum Requirements:
- NVIDIA GPU with CUDA Compute Capability 3.0+
- CUDA Toolkit installed
- Compatible drivers (check with `nvidia-smi`)
- Development environment:
  - Linux, WSL2, Windows(No debug support)
  - GCC/Clang/VS (depending on platform)


### Test Setup:
```bash
nvcc --version
nvidia-smi
```

---

## 11.8 NVIDIA Driver and CUDA Installation on Ubuntu 24.04

Follow these steps to install the NVIDIA CUDA Toolkit on Ubuntu 24.04 for GPU-accelerated computing.

### Step 1: Update System Packages

Ensure your system is up-to-date:

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install NVIDIA Drivers

Install the recommended NVIDIA drivers:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify the driver installation:

```bash
nvidia-smi
```

### Step 3: Add CUDA Repository

Download and install the CUDA repository keyring:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

Update the package list:

```bash
sudo apt update
```

### Step 4: Install CUDA Toolkit

Install the latest CUDA Toolkit:

```bash
sudo apt install -y cuda-toolkit
```

This will install all necessary components, including nvcc (CUDA compiler).

### Step 5: Configure Environment Variables

Add CUDA paths to your environment:

```bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Verify Installation

Check the installed CUDA version:

```bash
nvcc --version
```

Run nvidia-smi to confirm GPU and driver compatibility.

---

## 11.9 Installing Clang 20 on Ubuntu 24.04

Follow these steps to install Clang 20, the latest version of the LLVM compiler.

### Step 1: Add the LLVM Repository

The official LLVM repository provides the latest versions of Clang. Start by adding it to your system:

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20
```

This script automatically adds the repository, updates the package list, and installs Clang 20.

### Step 2: Verify Installation

After installation, confirm that Clang 20 is installed:

```bash
clang-20 --version
```

This should display the installed version as clang version 20.x.

### Step 3: Set Clang 20 as Default (Optional)

If you want Clang 20 to be the default compiler, update the symbolic links:

```bash
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100
```

Verify the default version:

```bash
clang --version
```

---

## 11.10 Installing Ubuntu on WSL2 (Windows Users)

For Windows users who want to run CUDA development on Ubuntu, you can install Ubuntu on WSL2 (Windows Subsystem for Linux).

### Quick Installation via Command Line

Open PowerShell or Windows Command Prompt in administrator mode and run:

```powershell
wsl --install -d Ubuntu-24.04
```

This command will:
- Enable required Windows features
- Install WSL2
- Download and install Ubuntu 24.04 LTS

### After Installation

1. **Set up your Ubuntu user account** when prompted
2. **Update packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. **Verify WSL2 is being used**:
   ```powershell
   wsl --list --verbose
   ```

### CUDA Support in WSL2

WSL2 supports CUDA through the Windows GPU driver. You don't need to install a GPU driver inside WSL2 - it uses the Windows NVIDIA driver.

To install CUDA Toolkit in WSL2:
```bash
sudo apt-key del 7fa2af80  # Remove old key if present
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit
```

### Reference

For detailed installation instructions and troubleshooting, see the official Ubuntu WSL documentation:
https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/