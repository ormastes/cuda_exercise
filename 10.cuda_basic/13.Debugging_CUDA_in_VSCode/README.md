# 🐞 Part 13: Debugging CUDA in VSCode

**Goal**: Master GPU debugging techniques using NVIDIA Nsight and cuda-gdb in VSCode.

## Project Structure
```
13.Debugging_CUDA_in_VSCode/
├── README.md            - Debugging guide and configuration
├── CMakeLists.txt       - Build configuration with debug flags
└── vector_add_2d.cu     - Example kernels for debugging practice
```

---

## **13.1 Overview of CUDA Debugging**

CUDA debugging allows you to:
- Set breakpoints inside GPU kernels
- Inspect thread-specific variables
- Step through kernel code line by line
- Examine memory access patterns
- Detect race conditions and memory errors

---

## **13.2 VSCode Debug Configuration**

### **13.2.1 Required Extensions**

* [x] **NVIDIA Nsight Visual Studio Code Edition** - Search for "Nsight" in VSCode Extensions marketplace
* [x] CMake Tools (for CMake integration)
* [x] C/C++ Extension Pack

**Key Features Provided by the Nsight Extension:**
- CUDA syntax highlighting and IntelliSense
- Kernel breakpoint support
- Conditional breakpoints for specific threads
- CPU and GPU debugging in same session
- Manual block and thread coordinate selection
- Variable and call-stack inspection

### **13.2.2 Debug Launch Configuration**

Create or update `.vscode/launch.json` in your workspace root to configure CUDA debugging:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      // GPU kernel debugging on Linux/WSL via Nsight VS Code (cuda-gdb)
      "name": "CUDA C++: Launch (cuda-gdb)",
      "type": "cuda-gdb",
      "request": "launch",
      "program": "${command:cmake.launchTargetPath}",
      "args": [],
      "cwd": "${workspaceFolder}",
      // Change it to local cuda-gdb
      "linux": {
        "debuggerPath": "/usr/bin/cuda-gdb"
      },
      "breakOnLaunch": false,
      "onAPIError": "stop"
    }
  ]
}
```

**Key Configuration Options:**
- `type: "cuda-gdb"`: Uses NVIDIA's CUDA debugger
- `program`: Path to executable (dynamically resolved by CMake)
- `debuggerPath`: Location of cuda-gdb binary on your system
- `breakOnLaunch`: Whether to pause at program start
- `onAPIError`: "stop" - Breaks execution when CUDA API errors occur

---

## **13.3 Build Configuration for Analysis**

### **13.3.1 CMakeLists.txt with Profiling Support**

**Updated Implementation from Root CMakeLists.txt:**

```cmake
# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-Gl-O0>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    # Profile flags: Optimized with debug info
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O2>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O2>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release flags: Optimize for performance
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O3;-use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O3>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
endif()

```

**Key Improvements:**
1. **Build Type Separation**: Different optimization levels for Debug, Profile, and Release builds
2. **Language-Specific Flags**: Separate compile options for CUDA and C++ code using generator expressions
3. **Debug Build (`-G -g -O0`)**: Full debugging symbols with no optimization for best debugging experience
4. **Profile Build (`-G -g -O2`)**: Balanced optimization with debug info for performance analysis
5. **Release Build (`-O3 -use_fast_math`)**: Maximum optimization with fast math for production performance
6. **Generator Expressions**: Use `$<$<COMPILE_LANGUAGE:CUDA>>` for language-specific flags

### **13.3.2 Building with Different Configurations Using Ninja**

```bash
# Debug build - Full debugging capabilities
mkdir build_debug && cd build_debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
ninja

# Profile build - For performance analysis with debug info
mkdir build_profile && cd build_profile
cmake -G Ninja -DCMAKE_BUILD_TYPE=Profile ..
ninja

# Release build - Optimized for production
mkdir build_release && cd build_release
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja

# Clean and rebuild
ninja clean && ninja

# Verbose build to see actual compiler commands
ninja -v
```

### **13.3.3 Compiler Flags Reference**

| Flag | Purpose | Debug | Profile | Release |
|------|---------|-------|---------|---------|
| `-G` | Generate debug info for device code | ✅ | ✅ | ❌ |
| `-g` | Generate debug info for host code | ✅ | ✅ | Optional |
| `-O0` | Disable all optimizations | ✅ | ❌ | ❌ |
| `-O2` | Moderate optimization level | ❌ | ✅ | ❌ |
| `-O3` | Maximum optimization level | ❌ | ❌ | ✅ |
| `-use_fast_math` | Enable fast math operations | ❌ | ❌ | ✅ |
| `-lineinfo` | Embed line number information | ❌ | ✅ | ❌ |
| `--ptxas-options=-v` | Show register and memory usage | Optional | ✅ | ❌ |

### **13.3.4 Performance Impact of Debug Flags**

**Debug vs Release Performance Comparison:**
- Debug builds can be **10-100x slower** than release builds
- The `-G` flag disables many GPU optimizations for debugging visibility
- Device debug info significantly impacts kernel execution time
- Host debug info (`-g`) has minimal runtime performance impact

**Build Configuration Selection Guide:**
| Scenario | Recommended Build | Rationale |
|----------|-------------------|-----------|
| **Bug Investigation** | Debug | Full variable visibility, accurate stepping |
| **Performance Profiling** | Profile | Optimized code with line-level correlation |
| **Production Deployment** | Release | Maximum performance, smallest binary size |
| **Unit Testing** | Debug | Better error messages and stack traces |
| **Benchmarking** | Release | Realistic performance measurements |

**Ninja Build System Advantages:**
- **Faster incremental builds** compared to Make
- **Better parallelization** of build tasks
- **Cleaner output** with progress indicators
- **Automatic dependency detection** for CUDA files

---

## **13.4 Debugging Workflow**

### **Prerequisites**
- Complete example code (`vector_add_2d.cu` in this directory)
- Working CMake build configuration
- VSCode with Nsight extension installed

### **Step-by-Step Debugging Process**

1. **Set Breakpoints in Kernel Code**
   - Click on the line number in your `.cu` file
   - Breakpoints work inside `__global__` and `__device__` functions

2. **Build in Debug Mode**
   ```bash
   # Configure CMake with debug flags
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make
   ```
   - The `-G` flag for device debugging is automatically added in debug mode
   - The `-g` flag for host debugging is also included

3. **Start Debugging Session**
   - Press `F5` or click "Run and Debug" in sidebar
   - Select "CUDA C++: Launch (cuda-gdb)" configuration
   - The debugger will launch and pause at main() or your first breakpoint

4. **Navigate Through GPU Threads**
   - Use the debug panel dropdown to switch between threads
   - View thread coordinates (threadIdx, blockIdx)

5. **Inspect Variables**
   - Hover over variables to see values (e.g., `x`, `y`, `i` in vector_add_2d)
   - Use Watch panel for complex expressions like `y * width + x`
   - Variables panel shows local and global values
   - Inspect 2D grid indices and device function results

---

## **13.5 Debug Features and Commands**

### **13.5.1 Breakpoint Types**

| Type | Description | Usage |
|------|-------------|-------|
| **Line Breakpoints** | Stops at specific line | Click on line number |
| **Conditional Breakpoints** | Stops when condition is true | Right-click → "Add Conditional Breakpoint" |
| **Function Breakpoints** | Stops at function entry | Add function name in Breakpoints panel |
| **Logpoints** | Logs message without stopping | Right-click → "Add Logpoint" |

**CUDA-Specific Breakpoint Behavior:**
- Breakpoints in CUDA kernels are **initially unconditional**
- They are **reset on each kernel launch**
- When a breakpoint is hit, the debugger automatically switches focus to that thread
- Multiple threads hitting the same breakpoint will all pause

### **13.5.2 Setting Conditional Breakpoints for CUDA**

To debug specific threads, use conditional breakpoints:

1. **Right-click** on the line number where you want to break
2. Select **"Add Conditional Breakpoint..."**
3. Enter a condition using CUDA built-in variables:

**Common CUDA Conditional Expressions (for vector_add_2d.cu's 2D grid):**
```cpp
// Break only for thread (0,0) in block (0,0) - useful for vector_add_2d
threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0

// Break for the last thread in a 2D block
threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1

// Break for specific 2D position in vector_add_2d
int x = blockIdx.x * blockDim.x + threadIdx.x; x == 512 && y == 512

// Break for boundary threads in 2D
threadIdx.x == 0 || threadIdx.y == 0

// Break for specific warp in reduce_sum
(threadIdx.x / 32) == 2
```

### **13.5.3 Debug Controls**

| Control | Shortcut | Description |
|---------|----------|-------------|
| Continue | `F5` | Resume execution |
| Step Over | `F10` | Execute current line |
| Step Into | `F11` | Enter function call |
| Step Out | `Shift+F11` | Exit current function |
| Restart | `Ctrl+Shift+F5` | Restart debugging |
| Stop | `Shift+F5` | End debug session |

---

## **13.6 Thread and Block Navigation**

### **13.6.1 CUDA Focus Panel**

The CUDA Focus panel in VSCode shows:
- Current thread coordinates (x, y, z)
- Current block coordinates (x, y, z)
- Warp information
- Lane ID within warp

### **13.6.2 Switching Between Threads - Multiple Methods**

**Important:** Thread switching allows you to inspect variables and execution state from different parallel threads.

#### **Method 1: Status Bar Thread Selection**

The Nsight VSCode extension provides manual block and thread coordinate selection:

1. **Status Bar Method:**
   - During debugging, the current thread/block is shown in the status bar
   - Click on the thread indicator to change focus
   - Select new block and thread coordinates from the picker

2. **Note:** The exact command name may vary by extension version. Look for:
   - Commands starting with "CUDA" in the command palette
   - Thread/block information in the status bar during debugging

#### **Method 2: Debug Console Commands**

Use cuda-gdb commands directly in the Debug Console:

1. Open Debug Console (View → Debug Console or `Ctrl+Shift+Y`)
2. Type cuda-gdb commands:
   ```bash
   cuda thread (x,y,z)    # Switch to specific thread
   cuda block (x,y,z)     # Switch to specific block
   info cuda threads      # List all threads
   info cuda blocks       # List all blocks
   info cuda kernels      # Show kernel information
   ```

#### **Method 3: Automatic Focus on Breakpoint**

The debugger can automatically change focus when different threads hit breakpoints:
- When a thread hits a breakpoint, focus automatically switches to that thread
- This allows debugging multiple threads without manual switching
- Useful for catching race conditions or thread-specific issues

### **13.6.3 Debugging Views in VSCode**

When debugging CUDA with Nsight VSCode Edition, you use VSCode's standard debugging panels:

1. **Variables Panel** - Shows local variables for the current CUDA thread
2. **Watch Panel** - Add expressions to evaluate in current thread context
3. **Call Stack Panel** - Shows the execution stack with current block/thread info
4. **Breakpoints Panel** - Manage all breakpoints including conditional ones

**Important Notes:**
- These are VSCode's standard debugging panels, not CUDA-specific views
- The panels update based on the current CUDA thread focus
- Variables show values specific to the currently focused thread
- Use the status bar or command palette to change thread focus

---

## **13.7 Common Debugging Scenarios**

### **13.7.1 Debugging Array Access**

**Example: Vector Addition 2D Kernel**
```cpp
// File: vector_add_2d.cu (example implementation)
__device__ float square(float x) {
    return x * x;
}

__global__ void vector_add_2d(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    // Set breakpoint here to inspect 2D index calculation
    if (x < width && y < height) {
        C[i] = square(A[i]) + B[i];  // Breakpoint: Check array values and square function
    }
}
```

**What to inspect:**
- Values of `x`, `y`, and `i` for different threads
- 2D to 1D index mapping calculation
- Array contents at `A[i]`, `B[i]`
- Boundary conditions when `x` approaches `width` and `y` approaches `height`
- Device function `square()` execution

### **13.7.2 Debugging Race Conditions**

```cpp
// File: vector_add_2d.cu (included in this directory)
__global__ void reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data to shared memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();  // Breakpoint: Check all threads reached here

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];  // Breakpoint: Verify reduction
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);  // Breakpoint: Check atomic operation
    }
}
```

### **13.7.3 Debugging Memory Errors**

Enable memory checking with cuda-memcheck:

```json
{
    "name": "CUDA Memory Check",
    "type": "cuda-gdb",
    "request": "launch",
    "program": "${command:cmake.launchTargetPath}",
    "args": [],
    "cwd": "${workspaceFolder}",
    "linux": {
        "debuggerPath": "/usr/bin/cuda-gdb",
        "MIMode": "gdb",
        "setupCommands": [
            {
                "text": "set cuda memcheck on"
            }
        ]
    }
}
```

---

## **13.8 Debug Output and Logging**

### **13.8.1 Using printf in Kernels**

**Note:** printf in kernels requires compute capability 2.0 or higher.

```cpp
// Adapting vector_add_2d.cu for printf debugging
__global__ void vector_add_2d(const float* A, const float* B, float* C, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    // Print from specific thread for debugging
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Block (%d,%d) started processing\n", blockIdx.x, blockIdx.y);
    }

    if (x < width && y < height) {
        // Print values for debugging specific threads
        if (x == 0 && y == 0) {  // First element
            printf("Thread (%d,%d,%d,%d) processing index %d, A=%f, B=%f\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, A[i], B[i]);
        }
        C[i] = square(A[i]) + B[i];
    }
}
```

### **13.8.2 Debug Console Output**

The Debug Console in VSCode shows:
- Program output (printf statements)
- CUDA API errors
- Kernel launch failures
- Memory allocation errors

---

## **13.9 Performance Debugging**

### **13.9.1 Identifying Performance Issues**

Use these techniques while debugging:

1. **Check Occupancy**: View active warps vs maximum warps
2. **Memory Access Patterns**: Identify coalesced vs uncoalesced access
3. **Branch Divergence**: Detect when threads in a warp take different paths
4. **Synchronization Points**: Find unnecessary __syncthreads()

### **13.9.2 Debug Build vs Release Build**

| Aspect | Debug Build | Release Build |
|--------|-------------|---------------|
| Compilation Flags | `-G -g` | `-O3` |
| Performance | Slower (no optimizations) | Faster (optimized) |
| Debugging | Full symbol information | Limited debugging |
| Memory Layout | Predictable | Optimized/reordered |

---

## **13.10 Troubleshooting Common Issues**

**Before debugging, ensure:**
- NVIDIA GPU drivers are installed and up to date
- CUDA Toolkit is properly installed
- cuda-gdb is accessible in your PATH or specified correctly in launch.json

### **Problem 1: Debugger Not Stopping at Breakpoints**

**Solutions:**
- Ensure code is compiled with `-G` flag for device debug info
- Check that breakpoint is in device code, not host code
- Verify the kernel is actually being executed

### **Problem 2: Cannot See Variable Values**

**Solutions:**
- Variable might be optimized out - use debug build
- Variable might be in register - check different threads
- Use `volatile` keyword to prevent optimization

### **Problem 3: Debugger Crashes or Hangs**

**Solutions:**
- Reduce number of threads/blocks for debugging
- Check for infinite loops in kernel
- Ensure CUDA toolkit version matches cuda-gdb version

---

## **13.11 Advanced Debugging Tips**

### **13.11.1 Using Guard Conditions for Thread-Specific Debugging**

Add guard conditions in your kernel to create targeted debug points:

```cpp
// In vector_add_2d.cu - Only break for specific thread in 2D grid
if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    // Set breakpoint here for 2D grid debugging
    int debug_point = 1;
    // Can examine x, y, i variables at this point
}
```

### **13.11.2 Warp-Level Debugging**

Debug at warp granularity:

```cpp
// Warp-level debugging in reduce_sum from vector_add_2d.cu
__global__ void reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    // Warp-level debugging
    int warpId = tid / 32;
    int laneId = tid % 32;

    // Break only for first thread of each warp
    if (laneId == 0) {
        printf("Warp %d in block %d starting reduction\n", warpId, blockIdx.x);
        // Set breakpoint here for warp debugging
    }

    // Rest of reduction code...
}
```

### **13.11.3 Memory Fence Debugging**

Debug memory consistency issues:
This may not happen on this example, but if you have memory synchronization issues, you can add memory fences to ensure visibility.

```cpp
// Memory synchronization in reduce_sum from vector_add_2d.cu
__global__ void reduce_sum(const float* input, float* output, int N) {
    // ... reduction code ...

    // Write result for this block to global memory
    if (tid == 0) {
        // Memory fence ensures all threads see updated sdata[0]
        __threadfence_block();  // Breakpoint: Check memory visibility
        atomicAdd(output, sdata[0]);  // Atomic operation for thread safety
    }
}
```

### **13.11.4 Debugging extern __shared__ Memory**

**⚠️ Important Limitation:** If you use `extern __shared__` memory, you will not see it as a named variable in CUDA debuggers like cuda-gdb.

**🔧 Why?**
Because `extern __shared__` declares a nameless shared memory array whose size is defined dynamically at kernel launch time, not in the source code.

**Example:**
```cpp
__global__ void myKernel() {
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = threadIdx.x;
}
```

Here, `shared_mem` is just a pointer to shared memory — the variable itself has no name in the compiled symbol table. CUDA debuggers like cuda-gdb or Nsight cannot display it as a named shared memory variable, and you won't find it under the shared scope in your debugger UI.

**Debugging Workarounds:**

1. **Use temporary variables to inspect values:**
```cpp
if (tid == 0) {
    float result = sdata[0];  // Force read into debuggable variable
    atomicAdd(output, result);
}
```

2. **Use volatile to prevent optimization:**
```cpp
if (tid == 0) {
    volatile float* vsdata = sdata;
    atomicAdd(output, vsdata[0]);  // Can now inspect vsdata[0]
}
```

3. **Use printf for debugging:**
```cpp
if (tid == 0) {
    printf("Block %d: sdata[0] = %f\n", blockIdx.x, sdata[0]);
    atomicAdd(output, sdata[0]);
}
```

4. **Compile with debug flags to minimize optimizations:**
```bash
nvcc -G -g -O0 vector_add_2d.cu -o vector_add_2d
```

---

## 🧠 **13.12 Debug Commands Reference**

**Note:** These commands are used in the VSCode Debug Console during an active debugging session.

### **cuda-gdb Commands (in Debug Console)**

#### **Basic Information Commands**
| Command | Description | Example Output |
|---------|-------------|----------------|
| `info cuda kernels` | List all running kernels | Shows kernel ID, grid dimensions, block dimensions |
| `info cuda blocks` | Show all blocks in current kernel | Lists block ranges and their states |
| `info cuda threads` | Show all threads with their locations | Shows thread ranges, PC, and source location |
| `info cuda warps` | Display warp information | Lists warps and their execution state |
| `info cuda sm` | Show current SM information | Displays streaming multiprocessor details |
| `info threads` | List CPU threads | Shows host-side threads |
| `backtrace` | Show call stack | Displays function call hierarchy |

#### **Thread Navigation Commands**
| Command | Description | Example |
|---------|-------------|---------|
| `cuda thread (x,y,z)` | Switch to specific thread in current block | `cuda thread 15,15,0` |
| `cuda block (x,y,z)` | Switch to specific block | `cuda block 1,0,0` |
| `cuda kernel <id> block <x,y,z> thread <x,y,z>` | Focus on specific thread (full syntax) | `cuda kernel 0 block 1,0,0 thread 15,0,0` |
| `cuda kernel block thread` | Show current focus | Displays current kernel, block, and thread |
| `cuda warp <lane_mask>` | Switch to specific warp | Navigate by warp ID |

#### **Variable Inspection**
| Command | Description | Example |
|---------|-------------|---------|
| `print threadIdx` | Show current thread index | `$1 = {x = 0, y = 0, z = 0}` |
| `print blockIdx` | Show current block index | `$2 = {x = 1, y = 0, z = 0}` |
| `print blockDim` | Show block dimensions | `$3 = {x = 16, y = 16, z = 1}` |
| `print gridDim` | Show grid dimensions | `$4 = {x = 64, y = 64, z = 1}` |
| `print <variable>` | Print any variable value | `print i` shows value for current thread |
| `info locals` | Show all local variables | Lists all variables in current scope |

#### **Execution Control**
| Command | Description |
|---------|-------------|
| `set cuda single-stepping on` | Step through only current thread |
| `set cuda scheduler single-stepping on` | Single-step one warp at a time |
| `break <kernel> if <condition>` | Conditional breakpoint | 
| `next` | Step to next line (current thread with single-stepping) |
| `continue` | Resume execution |

#### **Practical Examples**
```bash
# Set breakpoint for specific thread
(cuda-gdb) break vector_add_2d if blockIdx.x==0 && threadIdx.x==0

# Break when accessing specific array index
(cuda-gdb) break vector_add_2d if i==1000

# Navigate to specific thread and inspect
(cuda-gdb) cuda kernel 0 block 2,1,0 thread 5,3,0
(cuda-gdb) print x
(cuda-gdb) print y
(cuda-gdb) print i

# Check what threads are at breakpoint
(cuda-gdb) info cuda threads
```

**Note:** There is no `threadDim` variable in CUDA - use `blockDim` for thread dimensions per block and `gridDim` for block dimensions per grid.

### **13.12.1 Understanding cuda-gdb Output**

#### **`info cuda kernels` Output Explained**
```
(cuda-gdb) info cuda kernels
  Kernel Parent Dev Grid Status                 SMs Mask   GridDim  BlockDim Invocation
*      0      -   0    1 Active 0x0fffffffffffffffffffff (64,64,1) (16,16,1) vector_add_2d(...)
```

| Field | Value | Meaning |
|-------|-------|---------|
| **Kernel** | 0 | Kernel ID (asterisk * indicates current focus) |
| **Parent** | - | Parent kernel (- means no parent) |
| **Dev** | 0 | GPU device ID |
| **Grid** | 1 | Grid ID |
| **Status** | Active | Kernel execution state |
| **SMs Mask** | 0x0fff... | Streaming Multiprocessors mask (which SMs are used) |
| **GridDim** | (64,64,1) | Grid dimensions: 64×64 blocks |
| **BlockDim** | (16,16,1) | Block dimensions: 16×16 threads per block |
| **Invocation** | vector_add_2d(...) | Kernel name and parameters |

**Total threads**: 64×64 blocks × 16×16 threads = 1,048,576 threads

#### **`info cuda blocks` Output Explained**
```
(cuda-gdb) info cuda blocks
  BlockIdx To BlockIdx Count   State
Kernel 0
*  (0,0,0)    (55,7,0)   504 running
```

| Field | Value | Meaning |
|-------|-------|---------|
| **BlockIdx** | (0,0,0) | Starting block index |
| **To BlockIdx** | (55,7,0) | Ending block index range |
| **Count** | 504 | Number of blocks in this state |
| **State** | running | Execution state of these blocks |

**Interpretation**: 504 blocks from (0,0,0) to (55,7,0) are currently running

#### **`info cuda threads` Output Explained**
```
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx ThreadIdx Count         PC                    Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0) (15,15,0)   256 0x00007fffd527e220 vector_add_2d.cu     6
   (1,0,0)   (0,0,0)     (1,0,0)  (15,1,0)    32 0x00007fffd527d570 vector_add_2d.cu    15
```

| Field | Value | Meaning |
|-------|-------|---------|
| **BlockIdx** | (0,0,0) | Block coordinates |
| **ThreadIdx** | (0,0,0) | Starting thread in block |
| **To BlockIdx ThreadIdx** | (0,0,0) (15,15,0) | Ending thread range |
| **Count** | 256 | Number of threads at this PC |
| **PC** | 0x00007fffd527e220 | Program Counter (instruction address) |
| **Filename** | vector_add_2d.cu | Source file |
| **Line** | 6 | Line number in source |

**Key Insights**:
- Threads are grouped by their current execution point (PC)
- Different thread groups may be at different lines of code
- The asterisk (*) shows current focus
- Line 6 and 15 indicate different execution points in the kernel

#### **`info threads` (CPU Threads) Output Explained**
```
(cuda-gdb) info threads
  Id   Target Id                                            Frame
  1    Thread 0x7ffff7bcf000 (LWP 759870) "12_YourFirstCUD" 0x00007ffff0e61274 in ?? () from libcuda.so.1
  2    Thread 0x7ffff09ff000 (LWP 759874) "cuda00001400006" 0x00007ffff751b4cd in poll () from libc.so.6
```

| Thread | Purpose |
|--------|---------|
| **Thread 1** | Main application thread |
| **Thread 2-3** | CUDA runtime threads |
| **cuda-EvtHandlr** | CUDA event handler thread |

These are CPU-side threads, not GPU threads. They manage the CUDA runtime and kernel launches.

---

## ✅ **13.13 Summary**

**Getting Started:**
1. Install the Nsight VSCode Extension from the marketplace
2. Configure your launch.json with cuda-gdb settings
3. Build your CUDA project with debug flags (`-G` for device, `-g` for host)
4. Set breakpoints and start debugging with F5

**Key Takeaways:**
- CUDA debugging in VSCode uses cuda-gdb through Nsight extension
- You can set breakpoints and step through GPU kernel code
- Thread navigation allows inspecting different parallel execution paths
- Debug builds with `-G` flag enable full debugging capabilities for device.
- Debug builds with `-g` flag enable full debugging capabilities for host.
- Memory checking helps identify access violations and race conditions

**Best Practices:**
1. Start with small problem sizes when debugging
2. Use printf for quick checks, debugger for detailed analysis
3. Test with different thread/block configurations
4. Always check boundary conditions
5. Verify synchronization points

---

## 📚 **13.14 References and Resources**

### **Official Resources**
- **[Nsight VSCode Extension on Marketplace](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)**
  - Installation instructions
  - Feature list and requirements
  - User reviews and Q&A

- **[CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html)**
  - Complete cuda-gdb command reference
  - Debugging techniques and examples
  
- **[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)**
  - Comprehensive CUDA programming guide
  - Best practices and optimization tips
  - [CUDA Debugging Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html#debugging-guide)

### **Additional Debugging Tools**
- **[Nsight Systems](https://developer.nvidia.com/nsight-systems)** - System-wide performance analysis
- **[Nsight Compute](https://developer.nvidia.com/nsight-compute)** - Kernel profiling and optimization
- **cuda-memcheck** - Memory error detection tool (included with CUDA Toolkit)

### **Community Resources**
- **[NVIDIA Developer Forums](https://forums.developer.nvidia.com/)**
  - [Thread Focus in VSCode Discussion](https://forums.developer.nvidia.com/t/how-to-change-the-thread-focus-using-the-visual-studio-code-extension/288639/2)
- **[Stack Overflow CUDA Tag](https://stackoverflow.com/questions/tagged/cuda)**
- **[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)**

---

📄 **Next**: Continue with more advanced CUDA programming concepts

---
