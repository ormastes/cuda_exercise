
# ⚙️ Part 2: Your First CUDA Kernel

**Goal**: Introduce kernel syntax, compilation, launch, and debugging in VSCode.

---

## **2.1 Host vs Device Code Separation**

CUDA functions are separated by **execution location** and **call origin**:

| Qualifier      | Executes On | Callable From | Description |
|----------------|-------------|----------------|-------------|
| `__host__`     | CPU         | CPU            | Default for all C/C++ functions |
| `__device__`   | GPU         | GPU            | Used for helper functions inside kernels |
| `__global__`   | GPU         | CPU            | Defines a kernel callable from the host |

You use:
- `__global__` for launching kernels
- `__device__` for GPU-side helper logic

---

## **2.2 `__global__`, `__device__`, and `dim3` API**

#### ✅ `__global__` Kernel
Callable from host, executed on GPU:
```cpp
__global__ void myKernel(...) { ... }
````

#### ✅ `__device__` Function

Helper function usable **only** inside other device functions:

```cpp
__device__ float square(float x) {
    return x * x;
}
```

#### ✅ `dim3`: 1D/2D/3D Thread Organization

CUDA provides the `dim3` type to specify grid/block dimensions cleanly:

```cpp
dim3 blockDim(16, 16);  // 16x16 threads
dim3 gridDim((width+15)/16, (height+15)/16);
myKernel<<<gridDim, blockDim>>>(...);
```

You can access:

* `threadIdx.x`, `threadIdx.y`
* `blockIdx.x`, `blockIdx.y`
* `blockDim.x`, `blockDim.y`

This helps map GPU threads to 2D/3D data like images or matrices.

---

## **2.3 Launch Configuration with `dim3`**

Let’s use `dim3` to process data in 2D:

```cpp
// Launching a 2D grid of 16x16 blocks with 16x16 threads each
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((width + 15)/16, (height + 15)/16);

kernel2D<<<blocksPerGrid, threadsPerBlock>>>(...);
```

Inside the kernel, get your coordinates like this:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## **2.4 Updated Vector Add Example with `dim3`**

```cpp
// vector_add_2d.cu
#include <iostream>
#include <cuda_runtime.h>

__device__ float square(float x) {
    return x * x;
}

__global__ void vectorAdd2D(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int N = width * height;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15)/16, (height + 15)/16);
    vectorAdd2D<<<blocks, threads>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << h_C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

---

## 🛠️ 2.5 VSCode Preset Selection & Build Setup

#### ✅ Step 1: Install Required Extensions

* [x] CUDA Nsight for VSCode
* [x] CMake Tools (if using CMake)
* [x] C/C++ IntelliSense by Microsoft

---

#### ✅ Step 2: VSCode Folder Structure

```
.vscode/
├── launch.json
├── tasks.json
CMakeLists.txt (optional)
vector_add_2d.cu
```

---

#### ✅ Step 3: `tasks.json` (build task)

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build CUDA",
      "type": "shell",
      "command": "/usr/local/cuda/bin/nvcc",
      "args": ["-G", "-g", "vector_add_2d.cu", "-o", "vector_add"],
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

---

#### ✅ Step 4: `launch.json` (debug task)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "CUDA Launch",
      "type": "cuda-gdb",
      "request": "launch",
      "program": "${workspaceFolder}/vector_add",
      "args": [],
      "cwd": "${workspaceFolder}",
      "stopAtEntry": false
    }
  ]
}
```

---

#### ✅ Step 5: Set Build Preset (If using `CMakePresets.json`)

You can select build preset in bottom bar of VSCode or via command palette:

> `CMake: Select Build Preset`

Example:

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "cuda-debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    }
  ]
}
```

---

## 🐞 2.6 Debugging CUDA in VSCode

1. Set breakpoint in kernel (e.g., `C[i] = ...`)
2. Run `Ctrl+Shift+B` to build
3. Press `F5` to debug using **Nsight CUDA Debugger**
4. Switch thread using dropdown:
   ![Switch CUDA Thread](images/switch_cuda_thread.png)
5. View GPU variables:
   ![Inspect CUDA Variables](images/cuda_variables_panel.png)

---

### 🧠 API Recap

| API / Feature     | Description                              |
| ----------------- | ---------------------------------------- |
| `__global__`      | GPU kernel callable from host            |
| `__device__`      | GPU-only helper functions                |
| `dim3`            | Grid/block configuration type            |
| `cudaMalloc`      | Allocates GPU memory                     |
| `cudaMemcpy`      | Transfers memory between host and device |
| `cudaFree`        | Frees GPU memory                         |
| `-G` compile flag | Enables device-side debug info           |
| Nsight for VSCode | NVIDIA’s official GPU debugger extension |

---

### ✅ Summary

* You learned to write your first kernel using `__global__` and `__device__`
* `dim3` helps organize threads for 2D and 3D data
* You can build and debug CUDA in VSCode using `tasks.json` + `launch.json`
* Nsight debugger lets you set breakpoints, inspect values, and switch threads

📄 Next: **Part 3 – Thread Hierarchy in Practice**

```

---

Let me know if you’d like me to generate Part 3 — it will cover thread mapping for images and matrices using 1D/2D/3D blocks and warps.
```
