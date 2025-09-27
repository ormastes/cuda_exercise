# 🔍 Part 4: Code Inspection, Sanitization, and Profiling

**Goal**: Master CUDA code analysis, error detection, and performance profiling using NVIDIA's comprehensive tool suite.

---

## **4.1 Overview**

This part covers three critical aspects of CUDA development:
1. **Code Inspection** - Static analysis and compiler diagnostics
2. **Sanitization** - Runtime error detection and memory checking
3. **Profiling** - Performance analysis and optimization

### **Tool Evolution**
- **cuda-memcheck** → **Compute Sanitizer** (CUDA 11.6+)
- **nvprof** → **Nsight Systems** & **Nsight Compute**
- Modern tools provide more comprehensive analysis

---

## **4.2 Build Configuration for Analysis**

### **4.2.1 CMakeLists.txt with Profiling Support**

**Updated Implementation from Root CMakeLists.txt:**

```cmake
# Compiler flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O0>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    # Profile flags: Optimized with debug info
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O2>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O2>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart CUDA::nvtx3)
    option(ENABLE_PROFILING "Enable profiling instrumentation" ON)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release flags: Optimize for performance
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G;-O3;-use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CXX>:-g3;-O3>"
    )
    set(CUDA_BASIC_LIB CUDA::cudart)
    option(ENABLE_PROFILING "Enable profiling instrumentation" OFF)
endif()

# Additional profiling flags
if(ENABLE_PROFILING)
    function(add_profiling_targets TARGET_NAME)
        # Single combined target for all profiling and checking - continues on errors
        add_custom_target(${TARGET_NAME}_profile
            COMMAND ${CMAKE_COMMAND} -E echo "=== Creating output directory ==="
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
            COMMAND bash -c "PROFILE_DIR=${CMAKE_BINARY_DIR}/gen/${TARGET_NAME}_$(date +%Y%m%d_%H%M%S) && \
                    mkdir -p $$PROFILE_DIR && \
                    echo \"Profile output directory: $$PROFILE_DIR\" && \
                    echo \"=== Nsight Systems Profiling ===\" && \
                    (nsys profile -o $$PROFILE_DIR/nsys_report --stats=true -t cuda,nvtx,osrt $<TARGET_FILE:${TARGET_NAME}> || echo \"Nsight Systems completed or failed\") && \
                    echo \"=== Nsight Compute Profiling ===\" && \
                    echo \"Note: If you see ERR_NVGPUCTRPERM, run: ./setup_ncu_permissions.sh\" && \
                    (ncu --set full -o $$PROFILE_DIR/ncu_report $<TARGET_FILE:${TARGET_NAME}> || echo \"Nsight Compute skipped - may need permissions\") && \
                    echo \"=== Compute Sanitizer All Checks: synccheck, racecheck, memcheck===\" && \
                    (compute-sanitizer $<TARGET_FILE:${TARGET_NAME}> > $$PROFILE_DIR/sanitizer_report.txt 2>&1 || echo \"Compute sanitizer completed or failed\") && \
                    echo \"=== Profile complete. Results saved in: $$PROFILE_DIR ===\""
            DEPENDS ${TARGET_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running complete profiling and checking suite on ${TARGET_NAME} (continues on errors)"
        )
    endfunction()
else()
    function(add_profiling_targets TARGET_NAME)
        message(WARNING "ENABLE_PROFILING variable is not enabled")
    endfunction()
endif()

# Verbose PTX compilation (shows register usage)
option(VERBOSE_PTXAS "Show PTX compilation details" OFF)
if(VERBOSE_PTXAS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
endif()
```

**Key Improvements:**
- ✅ **Error resilience**: Commands continue even if individual tools fail
- ✅ **Organized output**: All results saved in timestamped folders under `gen/`
- ✅ **Permission guidance**: Helpful message for Nsight Compute permission errors
- ✅ **Unified sanitizer**: Single compute-sanitizer run checks all issues

**Current Project CMakeLists.txt:**

```cmake
project(14_CodeInspectionAndProfiling)

add_executable(${PROJECT_NAME} vector_add_2d.cu)
add_profiling_targets(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}
PRIVATE
    ${CUDA_BASIC_LIB}
)
```

### **4.2.2 Setting Up Nsight Compute Permissions**

Before profiling with Nsight Compute, you may need to configure GPU performance counter permissions. A helper script `setup_ncu_permissions.sh` is provided in the root directory:

```bash
# Run the setup script
./setup_ncu_permissions.sh

# Available options:
# 1. PERMANENT - Via NVIDIA module (Recommended, requires reboot)
# 2. TEMPORARY - Reload kernel module (Advanced, immediate effect)
# 3. TEMPORARY - Via perf_event_paranoid (Simple, immediate effect)
```

The script handles:
- Automatic detection of Linux distribution (Debian/Ubuntu vs RedHat/Fedora)
- Proper initrd rebuilding for permanent changes
- Ubuntu-specific NVIDIA module naming conventions
- Clear instructions for verification

### **4.2.3 Building with Different Configurations**

```bash
# Debug build (for cuda-gdb)
cmake -DCMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release -B build_release
cmake --build build_release

# Profile build (optimized with symbols)
cmake -DCMAKE_BUILD_TYPE=Profile -DENABLE_PROFILING=ON -DVERBOSE_PTXAS=ON -B build_profile
cmake --build build_profile

# Run profiling (results saved in gen/ folder with timestamps)
cmake --build build_profile --target 14_CodeInspectionAndProfiling_profile
# Or with Ninja directly:
ninja -C build_profile 14_CodeInspectionAndProfiling_profile
```

**Profile Output Structure:**
```
build_profile/gen/
├── 14_CodeInspectionAndProfiling_20250113_143052/
│   ├── nsys_report.nsys-rep      # Nsight Systems timeline
│   ├── nsys_report.sqlite        # SQLite database for analysis
│   ├── ncu_report.ncu-rep        # Nsight Compute kernel metrics
│   └── sanitizer_report.txt      # Compute Sanitizer results
└── 14_CodeInspectionAndProfiling_20250113_144523/
    └── ...                        # Next profiling run
```

### **4.2.4 Viewing and Analyzing Results**

#### **Command-Line Analysis**

**Nsight Systems CLI:**
```bash
# Navigate to results directory
cd build_profile/gen/14_CodeInspectionAndProfiling_20250113_143052/

# Generate summary statistics
nsys stats nsys_report.nsys-rep

# Export specific reports
nsys stats --report cuda_api_sum nsys_report.nsys-rep
nsys stats --report cuda_kernel_sum nsys_report.nsys-rep
nsys stats --report cuda_memory_sum nsys_report.nsys-rep
nsys stats --report nvtx_sum nsys_report.nsys-rep

# Export to various formats
nsys export --type sqlite --output analysis.db nsys_report.nsys-rep
nsys export --type json --output timeline.json nsys_report.nsys-rep
nsys export --type text --output summary.txt nsys_report.nsys-rep
```

**Nsight Compute CLI:**
```bash
# View kernel summary
ncu --import ncu_report.ncu-rep --print-summary per-kernel

# View specific metrics
ncu --import ncu_report.ncu-rep --print-metrics regex:.*throughput.*

# Generate detailed text report
ncu --import ncu_report.ncu-rep --print-details all > kernel_details.txt

# Export to CSV for analysis
ncu --import ncu_report.ncu-rep --csv --print-summary per-kernel > metrics.csv
```

**Compute Sanitizer Results:**
```bash
# View sanitizer output
cat sanitizer_report.txt

# Filter for specific error types
grep "ERROR" sanitizer_report.txt
grep "WARNING" sanitizer_report.txt
grep "race" sanitizer_report.txt
```

#### **GUI Tools**

**Nsight Systems GUI:**
```bash
# Linux with display
nsys-ui build_profile/gen/*/nsys_report.nsys-rep

# Remote analysis (copy file to local machine with GUI)
scp user@remote:~/project/build_profile/gen/*/nsys_report.nsys-rep ./
nsys-ui nsys_report.nsys-rep
```

**Nsight Compute GUI:**
```bash
# Linux with display
ncu-ui build_profile/gen/*/ncu_report.ncu-rep

# Remote analysis
scp user@remote:~/project/build_profile/gen/*/ncu_report.ncu-rep ./
ncu-ui ncu_report.ncu-rep
```

**Web-based Visualization (Alternative):**
```bash
# For systems without GUI, generate HTML reports
# Nsight Systems HTML export
nsys stats --format html --output report.html nsys_report.nsys-rep

# View in browser
python3 -m http.server 8000  # Serve locally
# Navigate to http://localhost:8000/report.html
```

#### **Quick Analysis Script**

Create `analyze_profile.sh` for automated analysis:
```bash
#!/bin/bash
# analyze_profile.sh - Quick profile analysis

PROFILE_DIR=$1
if [ -z "$PROFILE_DIR" ]; then
    # Find latest profile directory
    PROFILE_DIR=$(ls -td build_profile/gen/*/ | head -1)
fi

echo "Analyzing: $PROFILE_DIR"
cd "$PROFILE_DIR"

echo -e "\n=== Kernel Performance Summary ==="
nsys stats --report cuda_kernel_sum nsys_report.nsys-rep 2>/dev/null | head -20

echo -e "\n=== Top Time-Consuming Kernels ==="
ncu --import ncu_report.ncu-rep --print-summary per-kernel 2>/dev/null | grep -A 10 "Kernel Name"

echo -e "\n=== Sanitizer Issues ==="
if [ -f sanitizer_report.txt ]; then
    grep -E "ERROR|WARNING" sanitizer_report.txt | head -10 || echo "No issues found"
fi

echo -e "\n=== Memory Throughput ==="
ncu --import ncu_report.ncu-rep --print-metrics regex:dram.*throughput 2>/dev/null

echo -e "\nFull reports available in: $PROFILE_DIR"
```

#### **VSCode Integration**

For VSCode users, install these extensions for better visualization:
- **NVIDIA Nsight Systems** - View .nsys-rep files directly in VSCode
- **CSV Viewer** - Analyze exported metrics in CSV format

Open results in VSCode:
```bash
code build_profile/gen/14_CodeInspectionAndProfiling_*/
```

---

## **4.3 Code Inspection and Static Analysis**

### **4.3.1 NVCC Compiler Analysis**

#### **Register Usage Analysis**
```bash
# Show register usage per kernel
nvcc -Xptxas -v matrix_multiply.cu -o matrix_multiply

# Example output:
# ptxas info    : Function properties for _Z14matrixMultiplyPfS_S_i
# ptxas         : Used 32 registers, 348 bytes cmem[0]
```

#### **Generate and Inspect PTX**
```bash
# Generate PTX intermediate code
nvcc -ptx matrix_multiply.cu -o matrix_multiply.ptx

# Keep all intermediate files
nvcc -keep matrix_multiply.cu

# Files generated:
# *.cpp1.ii  - Preprocessed host code
# *.cpp4.ii  - Preprocessed device code
# *.ptx      - PTX assembly
# *.cubin    - Binary device code
```

### **4.3.2 Clang-Tidy for CUDA**

#### **Setup and Basic Usage**

**Generate Compilation Database:**
```bash
# Enable compilation database generation in CMake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build
# This creates build/compile_commands.json

# For better CUDA support, use clang as the compiler
cmake -DCMAKE_CUDA_COMPILER=clang++ \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -B build_clang
```

**Run clang-tidy:**
```bash
# Single file with compilation database (uses .clang-tidy file for checks)
clang-tidy -p build vector_add_2d.cu

# Specify different build directories
clang-tidy -p build-clang vector_add_2d.cu
clang-tidy -p build-gcc vector_add_2d.cu
clang-tidy -p build_profile vector_add_2d.cu

# All CUDA files in the project
find . -name "*.cu" -o -name "*.cuh" | xargs clang-tidy -p build

# Override checks from .clang-tidy file temporarily
clang-tidy -p build \
    --checks='-*,modernize-*,performance-*,readability-*,bugprone-*' \
    vector_add_2d.cu

# Apply fixes automatically (use with caution!)
clang-tidy -p build --fix vector_add_2d.cu

# Export fixes to a file for review
clang-tidy -p build --export-fixes=fixes.yaml vector_add_2d.cu

# Run on specific files from the current directory
cd 10.cuda_basic/14.Code\ Inspection\ and\ Profiling/
clang-tidy -p ../../build-clang vector_add_2d.cu
```

#### **VSCode Integration**

**1. Install Extensions:**
- C/C++ (Microsoft)

**2. VSCode Settings (.vscode/settings.json):**
```json
{
    // Microsoft C/C++ Extension settings
    "C_Cpp.codeAnalysis.clangTidy.enabled": true,
    "C_Cpp.codeAnalysis.clangTidy.useBuildPath": true,
    "C_Cpp.codeAnalysis.clangTidy.path": "clang-tidy",
    "C_Cpp.codeAnalysis.clangTidy.args": [
        "-p",
        "${command:cmake.buildDirectory}"
    ],
    // Let .clang-tidy file handle the checks configuration
    "C_Cpp.codeAnalysis.runAutomatically": true,
    "C_Cpp.configurationWarnings": "disabled",
}
```

**Alternative: Use current CMake build directory dynamically:**
```json
{
    // Or specify the exact build directory
    "C_Cpp.codeAnalysis.clangTidy.args": [
        "-p",
        "${workspaceFolder}/build-clang"  // Use specific build directory
    ]
}
```

**3. Project .clang-tidy Configuration:**
```yaml
# .clang-tidy in project root
Checks: >
  -*,
  modernize-*,
  performance-*,
  readability-*,
  bugprone-*,
  cppcoreguidelines-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic

WarningsAsErrors: ''
HeaderFilterRegex: '.*\.(cu|cuh|hpp|h)$'
FormatStyle: file
CheckOptions:
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.GlobalConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.MacroDefinitionCase
    value: UPPER_CASE
  - key: performance-for-range-copy.WarnOnAllAutoCopies
    value: true
  - key: modernize-use-nullptr.NullMacros
    value: 'NULL'
```

**4. Run clang-tidy in VSCode:**
- **Automatic**: Errors/warnings appear in Problems panel as you type
- **Manual**: Ctrl+Shift+P → "Run Code Analysis on Active File"
- **Quick Fix**: Click on squiggly lines → lightbulb icon → apply fix

**5. Pre-commit Hook (optional):**
```bash
#!/bin/bash
# .git/hooks/pre-commit
# Run clang-tidy on staged CUDA files

FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cu|cuh)$')
if [ -n "$FILES" ]; then
    echo "Running clang-tidy on staged files..."
    for file in $FILES; do
        clang-tidy -p build "$file"
        if [ $? -ne 0 ]; then
            echo "clang-tidy found issues in $file"
            exit 1
        fi
    done
fi
```

#### **Common CUDA-Specific Checks**

```bash
# CUDA-aware clang-tidy checks
clang-tidy -p build \
    --checks='cuda-*,performance-no-int-to-ptr,modernize-use-override' \
    vector_add_2d.cu

# Custom query for CUDA patterns
clang-query -p build vector_add_2d.cu \
    -c "match cudaKernelCallExpr()"
```

### **4.3.3 CUDA Binary Utilities**

```bash
# Disassemble CUDA binary
cuobjdump -sass matrix_multiply

# List kernels in binary
cuobjdump -symbols matrix_multiply | grep STT_FUNC

# Extract PTX from fatbinary
cuobjdump -ptx matrix_multiply
```

---

## **4.4 Runtime Sanitization with Compute Sanitizer**

### **4.4.1 Memory Check (memcheck)**

Detects memory access errors:
- Out-of-bounds accesses
- Misaligned accesses
- Use of uninitialized memory
- Memory leaks

```bash
# Basic memory check
compute-sanitizer --tool memcheck ./program

# Save output to file
compute-sanitizer --tool memcheck --log-file memcheck.log ./program

# Check for leaks only
compute-sanitizer --tool memcheck --leak-check full ./program
```

**Example Output:**
```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x000001a0 in kernel_function(float*, int)
=========     by thread (31,0,0) in block (1,0,0)
=========     Address 0x7f3c89200800 is out of bounds
```

### **4.4.2 Race Condition Check (racecheck)**

Detects data races in shared memory:

```bash
# Check for race conditions
compute-sanitizer --tool racecheck ./program

# With hazard detection
compute-sanitizer --tool racecheck --racecheck-report hazard ./program
```

**Common Race Conditions Detected:**
- Write-after-write (WAW)
- Write-after-read (WAR)
- Read-after-write (RAW)

### **4.4.3 Synchronization Check (synccheck)**

Detects synchronization errors:

```bash
# Check synchronization issues
compute-sanitizer --tool synccheck ./program
```

**Detects:**
- Illegal __syncthreads() in conditional code
- Barrier divergence within warp
- Missing synchronization

### **4.4.4 Initialization Check (initcheck)**

Detects use of uninitialized memory:

```bash
# Check for uninitialized memory usage
compute-sanitizer --tool initcheck ./program
```

### **4.4.5 Example: Detecting Common Errors**

**File: `memory_errors.cu` - Example with intentional errors for sanitizer testing**

This example is included in the project. Key sections from the file:

```cuda
// memory_errors.cu - Example with intentional errors for sanitizer testing
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

__global__ void buggyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Error 1: Out-of-bounds access
    if (idx <= n) {  // BUG: Should be idx < n
        data[idx] = idx * 2.0f;
    }

    // Error 2: Race condition in shared memory
    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[idx];
    // BUG: Missing __syncthreads() here!

    // Use shared memory (will have race condition)
    if (threadIdx.x < 255) {
        data[idx] = sdata[threadIdx.x] + sdata[threadIdx.x + 1];
    }

    // Error 3: Conditional syncthreads (potential deadlock)
    if (threadIdx.x < 128) {
        __syncthreads();  // BUG: Not all threads reach this
    }
}

int main() {
    const int N = 1024;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *d_data;
    size_t size = N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_data, size));

    // Launch kernel with bugs
    buggyKernel<<<numBlocks, blockSize>>>(d_data, N);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_data));

    printf("Program completed (but may have errors!)\n");
    return 0;
}
```

**Compile and run with sanitizers:**
```bash
# Compile
nvcc -g -G memory_errors.cu -o memory_errors

# Detect out-of-bounds access
compute-sanitizer --tool memcheck ./memory_errors

# Detect race conditions
compute-sanitizer --tool racecheck ./memory_errors

# Detect synchronization issues
compute-sanitizer --tool synccheck ./memory_errors

# Run all checks
compute-sanitizer ./memory_errors
```

---

## **4.5 Performance Profiling with Nsight Systems**

### **4.5.1 System-Wide Timeline Profiling**

Nsight Systems provides system-level performance analysis:

#### **Basic Profiling**
```bash
# Generate profile with statistics
nsys profile --stats=true --output=report ./program

# Profile specific APIs
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o report ./program

# Profile with sampling
nsys profile --sample=cpu --sampling-frequency=1000 -o report ./program
```

#### **Viewing Results**
```bash
# Generate text report
nsys stats report.nsys-rep

# Export to SQLite for analysis
nsys export --sqlite report.sqlite report.nsys-rep

# Open GUI (if available)
nsys-ui report.nsys-rep
```

### **4.5.2 NVTX Markers for Custom Profiling**

**Complete runnable example with NVTX markers:**

```cuda
// nvtx_example.cu - Demonstrates NVTX profiling markers
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <chrono>
#include <thread>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Simple kernel for matrix multiplication
__global__ void matrixMulKernel(float* C, const float* A, const float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Vector addition kernel
__global__ void vectorAddKernel(float* c, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void initializeData(float* data, int size, float value) {
    for (int i = 0; i < size; i++) {
        data[i] = value + (i % 100) * 0.01f;
    }
}

int main() {
    const int N = 512;  // Matrix/vector size
    const int MATRIX_SIZE = N * N;
    const int VECTOR_SIZE = N * 1024;

    // Create main NVTX range for the entire program
    nvtxRangePushA("Main Program");

    // Memory allocation with NVTX markers
    nvtxRangePushA("Memory Allocation");
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices
    float *h_vec_a, *h_vec_b, *h_vec_c;  // Host vectors
    float *d_vec_a, *d_vec_b, *d_vec_c;  // Device vectors

    // Allocate host memory
    h_A = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_B = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_C = (float*)malloc(MATRIX_SIZE * sizeof(float));
    h_vec_a = (float*)malloc(VECTOR_SIZE * sizeof(float));
    h_vec_b = (float*)malloc(VECTOR_SIZE * sizeof(float));
    h_vec_c = (float*)malloc(VECTOR_SIZE * sizeof(float));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, MATRIX_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_a, VECTOR_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_b, VECTOR_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_vec_c, VECTOR_SIZE * sizeof(float)));
    nvtxRangePop();  // End Memory Allocation

    // Initialize data with NVTX marker
    nvtxRangePushA("Data Initialization");
    initializeData(h_A, MATRIX_SIZE, 1.0f);
    initializeData(h_B, MATRIX_SIZE, 2.0f);
    initializeData(h_vec_a, VECTOR_SIZE, 1.0f);
    initializeData(h_vec_b, VECTOR_SIZE, 2.0f);
    nvtxRangePop();  // End Data Initialization

    // Copy data to device with colored NVTX marker
    nvtxEventAttributes_t copyAttrib = {0};
    copyAttrib.version = NVTX_VERSION;
    copyAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    copyAttrib.colorType = NVTX_COLOR_ARGB;
    copyAttrib.color = 0xFFFF0000;  // Red
    copyAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    copyAttrib.message.ascii = "Host to Device Copy";

    nvtxRangePushEx(&copyAttrib);
    CHECK_CUDA(cudaMemcpy(d_A, h_A, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vec_a, h_vec_a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vec_b, h_vec_b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();  // End Host to Device Copy

    // Matrix multiplication with NVTX marker
    nvtxEventAttributes_t matrixAttrib = {0};
    matrixAttrib.version = NVTX_VERSION;
    matrixAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    matrixAttrib.colorType = NVTX_COLOR_ARGB;
    matrixAttrib.color = 0xFF00FF00;  // Green
    matrixAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    matrixAttrib.message.ascii = "Matrix Multiplication";

    nvtxRangePushEx(&matrixAttrib);
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMulKernel<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();  // End Matrix Multiplication

    // Vector addition with NVTX marker
    nvtxEventAttributes_t vectorAttrib = {0};
    vectorAttrib.version = NVTX_VERSION;
    vectorAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    vectorAttrib.colorType = NVTX_COLOR_ARGB;
    vectorAttrib.color = 0xFF0000FF;  // Blue
    vectorAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    vectorAttrib.message.ascii = "Vector Addition";

    nvtxRangePushEx(&vectorAttrib);
    int threadsPerBlock = 256;
    int blocksPerGrid = (VECTOR_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Multiple kernel launches to show in timeline
    for (int i = 0; i < 5; i++) {
        char iterName[32];
        sprintf(iterName, "Vector Add Iteration %d", i);
        nvtxRangePushA(iterName);
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec_c, d_vec_a, d_vec_b, VECTOR_SIZE);
        CHECK_CUDA(cudaDeviceSynchronize());
        nvtxRangePop();
    }
    nvtxRangePop();  // End Vector Addition

    // Copy results back with NVTX marker
    nvtxRangePushA("Device to Host Copy");
    CHECK_CUDA(cudaMemcpy(h_C, d_C, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_vec_c, d_vec_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();  // End Device to Host Copy

    // Cleanup with NVTX marker
    nvtxRangePushA("Memory Cleanup");
    free(h_A); free(h_B); free(h_C);
    free(h_vec_a); free(h_vec_b); free(h_vec_c);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_vec_a));
    CHECK_CUDA(cudaFree(d_vec_b));
    CHECK_CUDA(cudaFree(d_vec_c));
    nvtxRangePop();  // End Memory Cleanup

    nvtxRangePop();  // End Main Program

    printf("Program completed successfully!\n");
    printf("Run with: nsys profile --stats=true --trace=cuda,nvtx ./nvtx_example\n");
    return 0;
}
```

**Compile and profile:**
```bash
# Compile with NVTX support
nvcc -g -lineinfo nvtx_example.cu -lnvToolsExt -o nvtx_example

# Profile with Nsight Systems
nsys profile --stats=true --trace=cuda,nvtx,osrt -o nvtx_report ./nvtx_example

# View NVTX summary
nsys stats --report nvtx_sum nvtx_report.nsys-rep

# Open in GUI to see colored timeline
nsys-ui nvtx_report.nsys-rep
```

### **4.5.3 Command-Line Analysis**

```bash
# Profile with detailed CUDA metrics
nsys profile --stats=true --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --cuda-um-cpu-page-faults=true \
    --cuda-um-gpu-page-faults=true \
    -o detailed_report ./program

# Generate summary statistics
nsys stats --report cuda_api_sum detailed_report.nsys-rep
nsys stats --report cuda_kernel_sum detailed_report.nsys-rep
nsys stats --report cuda_memory_sum detailed_report.nsys-rep
```

---

## **4.6 Kernel Profiling with Nsight Compute**

### **4.6.1 Detailed Kernel Analysis**

Nsight Compute provides in-depth kernel performance analysis:

#### **Basic Profiling**
```bash
# Profile all kernels with default metrics
ncu ./program

# Profile with full metric set
ncu --set full ./program

# Profile specific kernel
ncu --kernel-name matrixMultiply --launch-skip 0 --launch-count 1 ./program

# Save report
ncu --set full -o kernel_analysis ./program
```

#### **Metric Sets**
```bash
# Available metric sets
ncu --list-sets

# Common sets:
# - basic: Essential metrics
# - full: Comprehensive analysis
# - detailed: Everything including source
# - roofline: Roofline model analysis
```

### **4.6.2 Specific Metric Collection**

```bash
# Memory throughput analysis
ncu --metrics dram_throughput,l2_cache_throughput,shared_throughput ./program

# Occupancy analysis
ncu --metrics sm_occupancy,achieved_occupancy,theoretical_occupancy ./program

# Warp efficiency
ncu --metrics warp_execution_efficiency,branch_efficiency ./program

# Register usage
ncu --metrics registers_per_thread,static_shared_memory_per_block ./program

# Memory access patterns
ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions ./program
```

### **4.6.3 Roofline Analysis**

```bash
# Generate roofline model
ncu --set roofline -o roofline_report ./program

# View in CLI
ncu --import roofline_report.ncu-rep --page raw
```

The roofline model shows:
- Arithmetic intensity (FLOPS/byte)
- Memory vs compute bound regions
- Optimization potential

### **4.6.4 Source-Level Analysis**

```bash
# Include source correlation
ncu --set full --target-processes all \
    --kernel-name matrixMultiply \
    --export source_analysis \
    ./program

# Generate with inline PTX/SASS
ncu --set full --print-source=sass ./program
```

### **4.6.5 Automated Performance Analysis**

```bash
# Generate optimization recommendations
ncu --set full --print-summary per-kernel \
    --export recommendations \
    ./program

# Focus on specific bottlenecks
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    ./program
```

---

## **4.7 Common Performance Bottlenecks**

### **4.7.1 Memory Bottlenecks**

#### **Detecting Memory Issues**
```bash
# Check memory throughput
ncu --metrics dram_read_throughput,dram_write_throughput ./program

# Check coalescing
ncu --metrics gld_efficiency,gst_efficiency ./program

# Check cache usage
ncu --metrics l1_cache_hit_rate,l2_cache_hit_rate ./program
```

#### **Example: Memory Coalescing with vector_add_2d.cu**

The project includes `vector_add_2d.cu` which demonstrates:
1. **NVTX Markers** - Visual profiling ranges
2. **Coalesced vs Strided Access** - Performance comparison
3. **Memory Error Detection** - Intentional bugs for sanitizer testing

**Running the Examples:**

```bash
# Build the project
cmake -DCMAKE_BUILD_TYPE=Profile -B build_profile
cmake --build build_profile

# Run with coalesced access (default)
./build_profile/10.cuda_basic/14.Code\ Inspection\ and\ Profiling/14_CodeInspectionAndProfiling

# Run with strided access comparison
./build_profile/10.cuda_basic/14.Code\ Inspection\ and\ Profiling/14_CodeInspectionAndProfiling --strided

# Run with memory error example (for sanitizer testing)
./build_profile/10.cuda_basic/14.Code\ Inspection\ and\ Profiling/14_CodeInspectionAndProfiling --memory-error
```

**Profiling the Examples:**

```bash
# Profile with NVTX visualization
nsys profile --trace=cuda,nvtx -o coalesced_report ./14_CodeInspectionAndProfiling
nsys profile --trace=cuda,nvtx -o strided_report ./14_CodeInspectionAndProfiling --strided

# Compare memory coalescing efficiency
ncu --metrics gld_efficiency,gst_efficiency --kernel-name ".*Coalesced" ./14_CodeInspectionAndProfiling
ncu --metrics gld_efficiency,gst_efficiency --kernel-name ".*Strided" ./14_CodeInspectionAndProfiling --strided

# Detect memory errors
compute-sanitizer ./14_CodeInspectionAndProfiling --memory-error
```

**Key Kernels in vector_add_2d.cu:**

1. **vectorAdd2D_Coalesced**: Row-major access pattern (good coalescing)
   - Adjacent threads access adjacent memory locations
   - High memory efficiency (~95% typical)

2. **vectorAdd2D_Strided**: Column-major access in row-major storage (poor coalescing)
   - Adjacent threads access strided memory locations
   - Low memory efficiency (~20-30% typical)

3. **vectorAdd2D_WithBug**: Missing boundary checks (for error detection)
   - Demonstrates out-of-bounds access
   - Use with compute-sanitizer to detect errors

4. **reduceSum**: Shared memory reduction with atomic operations
   - Shows shared memory usage patterns
   - Demonstrates synchronization requirements

**NVTX Ranges in the Code:**

The example uses hierarchical NVTX ranges to visualize different phases:
- Memory allocation (Host and Device)
- Data initialization
- Host-to-Device transfers (marked in red)
- Kernel execution (marked in green/blue)
- Device-to-Host transfers
- Cleanup operations

View in Nsight Systems GUI to see the colored timeline:
```bash
nsys-ui coalesced_report.nsys-rep
```

#### **Example: Alternative Memory Coalescing Comparison**

**Complete runnable example comparing coalesced vs uncoalesced access:**

```cuda
// memory_coalescing.cu - Compare memory access patterns
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// Poor memory access pattern - strided access
__global__ void uncoalescedAccess(float* data, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = tid * stride;  // Strided access - poor coalescing
        if (idx < n * stride) {
            data[idx] = idx * 2.0f;
        }
    }
}

// Good memory access pattern - sequential access
__global__ void coalescedAccess(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Sequential access
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

// Optimized with shared memory for data reuse
__global__ void optimizedAccess(float* output, const float* input, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Coalesced read to shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();

    // Process in shared memory (example: simple smoothing)
    if (tid > 0 && tid < blockDim.x - 1 && idx < n) {
        float result = 0.25f * sdata[tid - 1] +
                      0.5f * sdata[tid] +
                      0.25f * sdata[tid + 1];
        output[idx] = result;
    } else if (idx < n) {
        output[idx] = sdata[tid];
    }
}

float measureKernelTime(void (*kernel)(float*, int, int),
                       float* d_data, int size, int stride,
                       int blocks, int threads) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm up
    kernel<<<blocks, threads>>>(d_data, stride, size);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Measure
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        kernel<<<blocks, threads>>>(d_data, stride, size);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / 100.0f;  // Average time
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const int STRIDE = 32;      // Stride for uncoalesced access
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    printf("=== Memory Coalescing Performance Test ===\n");
    printf("Array size: %d elements\n", N);
    printf("Block size: %d threads\n", THREADS);
    printf("Grid size: %d blocks\n", BLOCKS);
    printf("Stride for uncoalesced: %d\n\n", STRIDE);

    // Allocate memory
    float *d_data1, *d_data2, *d_input, *d_output;
    size_t size = N * sizeof(float);
    size_t strided_size = N * STRIDE * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_data1, strided_size));
    CHECK_CUDA(cudaMalloc(&d_data2, size));
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    // Initialize data
    CHECK_CUDA(cudaMemset(d_data1, 0, strided_size));
    CHECK_CUDA(cudaMemset(d_data2, 0, size));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Test 1: Uncoalesced access
    printf("Testing uncoalesced access (strided)...\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        uncoalescedAccess<<<BLOCKS, THREADS>>>(d_data1, STRIDE, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float uncoalesced_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&uncoalesced_time, start, stop));
    uncoalesced_time /= 100.0f;

    // Test 2: Coalesced access
    printf("Testing coalesced access (sequential)...\n");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        coalescedAccess<<<BLOCKS, THREADS>>>(d_data2, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coalesced_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&coalesced_time, start, stop));
    coalesced_time /= 100.0f;

    // Test 3: Optimized with shared memory
    printf("Testing optimized access (shared memory)...\n");
    size_t shared_mem_size = THREADS * sizeof(float);
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        optimizedAccess<<<BLOCKS, THREADS, shared_mem_size>>>(d_output, d_input, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float optimized_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&optimized_time, start, stop));
    optimized_time /= 100.0f;

    // Calculate bandwidth
    float gb_transferred = (N * sizeof(float) * 2) / (1024.0f * 1024.0f * 1024.0f); // Read + Write

    printf("\n=== Results ===\n");
    printf("Uncoalesced access time: %.3f ms\n", uncoalesced_time);
    printf("Coalesced access time:   %.3f ms (%.2fx faster)\n",
           coalesced_time, uncoalesced_time / coalesced_time);
    printf("Optimized access time:   %.3f ms (%.2fx faster)\n",
           optimized_time, uncoalesced_time / optimized_time);

    printf("\n=== Effective Bandwidth ===\n");
    printf("Uncoalesced: %.2f GB/s\n", gb_transferred / (uncoalesced_time / 1000.0f));
    printf("Coalesced:   %.2f GB/s\n", gb_transferred / (coalesced_time / 1000.0f));
    printf("Optimized:   %.2f GB/s\n", gb_transferred / (optimized_time / 1000.0f));

    // Cleanup
    CHECK_CUDA(cudaFree(d_data1));
    CHECK_CUDA(cudaFree(d_data2));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\n=== Profiling Instructions ===\n");
    printf("To analyze memory patterns:\n");
    printf("1. ncu --metrics gld_efficiency,gst_efficiency ./memory_coalescing\n");
    printf("2. ncu --set detailed ./memory_coalescing\n");
    printf("3. nsys profile --stats=true ./memory_coalescing\n");

    return 0;
}
```

**Compile and analyze:**
```bash
# Compile
nvcc -O2 -lineinfo memory_coalescing.cu -o memory_coalescing

# Run the program
./memory_coalescing

# Profile memory efficiency
ncu --metrics gld_efficiency,gst_efficiency,gld_transactions,gst_transactions ./memory_coalescing

# Detailed analysis
ncu --set full --kernel-name ".*coalesced.*" ./memory_coalescing
```

### **4.7.2 Occupancy Issues**

#### **Analyzing Occupancy**
```bash
# Check occupancy metrics
ncu --metrics achieved_occupancy,theoretical_occupancy,sm_occupancy ./program

# Detailed occupancy analysis
ncu --section Occupancy ./program
```

#### **Improving Occupancy**
```cuda
// Configuration for better occupancy
void launchKernel(float* data, int n) {
    int blockSize;
    int minGridSize;

    // Calculate optimal block size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                        myKernel, 0, 0);

    int gridSize = (n + blockSize - 1) / blockSize;

    myKernel<<<gridSize, blockSize>>>(data, n);
}
```

### **4.7.3 Divergent Execution**

#### **Detecting Branch Divergence**
```bash
# Check branch efficiency
ncu --metrics branch_efficiency,divergent_branch ./program

# Warp execution efficiency
ncu --metrics warp_execution_efficiency ./program
```

#### **Example: Reducing Divergence**
```cuda
// Divergent code
__global__ void divergentKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {  // Divergent within warp
            data[idx] = compute_even(idx);
        } else {
            data[idx] = compute_odd(idx);
        }
    }
}

// Reduced divergence
__global__ void improvedKernel(float* even_data, float* odd_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_n = n / 2;

    // Process even indices
    if (idx < half_n) {
        even_data[idx] = compute_even(idx * 2);
    }

    // Process odd indices
    if (idx < half_n) {
        odd_data[idx] = compute_odd(idx * 2 + 1);
    }
}
```

---

## **4.8 Profiling Workflow**

### **4.8.1 Step-by-Step Optimization Process**

1. **Baseline Measurement**
   ```bash
   # System-level profile
   nsys profile --stats=true -o baseline ./program

   # Kernel-level profile
   ncu --set full -o baseline_kernels ./program
   ```

2. **Identify Bottlenecks**
   ```bash
   # Find slowest kernels
   nsys stats --report cuda_kernel_sum baseline.nsys-rep

   # Analyze specific kernel
   ncu --import baseline_kernels.ncu-rep --print-summary per-kernel
   ```

3. **Detailed Analysis**
   ```bash
   # Focus on problematic kernel
   ncu --kernel-name slowKernel --set full -o detailed ./program
   ```

4. **Apply Optimizations**
   - Fix memory access patterns
   - Improve occupancy
   - Reduce divergence
   - Use shared memory

5. **Verify Improvements**
   ```bash
   # Compare with baseline
   ncu --set full -o optimized ./program_optimized

   # Generate comparison
   ncu --import baseline_kernels.ncu-rep --import optimized.ncu-rep \
       --print-summary per-kernel
   ```

### **4.8.2 Automated Profiling Script**

```bash
#!/bin/bash
# profile_all.sh - Comprehensive profiling script

PROGRAM="./my_cuda_app"
OUTPUT_DIR="profiling_results"

mkdir -p $OUTPUT_DIR

echo "=== Running Sanitizers ==="
compute-sanitizer --tool memcheck $PROGRAM > $OUTPUT_DIR/memcheck.log 2>&1
compute-sanitizer --tool racecheck $PROGRAM > $OUTPUT_DIR/racecheck.log 2>&1
compute-sanitizer --tool synccheck $PROGRAM > $OUTPUT_DIR/synccheck.log 2>&1

echo "=== System Profiling ==="
nsys profile --stats=true --trace=cuda,nvtx,osrt \
    -o $OUTPUT_DIR/system_profile $PROGRAM

echo "=== Kernel Profiling ==="
ncu --set full -o $OUTPUT_DIR/kernel_profile $PROGRAM

echo "=== Generating Reports ==="
nsys stats --report cuda_kernel_sum \
    $OUTPUT_DIR/system_profile.nsys-rep > $OUTPUT_DIR/kernel_summary.txt

ncu --import $OUTPUT_DIR/kernel_profile.ncu-rep \
    --print-summary per-kernel > $OUTPUT_DIR/kernel_analysis.txt

echo "Profiling complete. Results in $OUTPUT_DIR/"
```

---

## **4.9 Best Practices**

### **4.9.1 Development Workflow**

1. **During Development**
   - Use sanitizers after every major change
   - Profile with small datasets first
   - Focus on algorithmic improvements before micro-optimizations

2. **Pre-Release**
   - Run full sanitizer suite
   - Profile with production-size datasets
   - Compare against performance targets

3. **Production Monitoring**
   - Use lightweight profiling (nsys with sampling)
   - Monitor key metrics
   - Set up automated performance regression tests

### **4.9.2 Common Pitfalls to Avoid**

1. **Profiling Debug Builds**
   - Always profile optimized builds
   - Use `-lineinfo` for source correlation

2. **Ignoring Sanitizer Warnings**
   - Fix all sanitizer issues before profiling
   - Memory errors can cause misleading performance data

3. **Over-Optimizing**
   - Focus on actual bottlenecks
   - Measure impact of each optimization
   - Consider code maintainability

### **4.9.3 Performance Targets**

| Metric | Good | Excellent |
|--------|------|-----------|
| Memory Bandwidth Utilization | >60% | >80% |
| Achieved Occupancy | >50% | >75% |
| Warp Execution Efficiency | >70% | >90% |
| Global Memory Load Efficiency | >80% | >95% |
| Branch Efficiency | >85% | >95% |

---

## **4.10 Advanced Topics**

### **4.10.1 Multi-GPU Profiling**

```bash
# Profile all GPUs
ncu --target-processes all --device all ./multi_gpu_program

# Profile specific GPU
CUDA_VISIBLE_DEVICES=0 ncu --set full ./program
```

### **4.10.2 MPI + CUDA Profiling**

```bash
# Profile MPI+CUDA application
mpirun -np 4 nsys profile -o report_%q{OMPI_COMM_WORLD_RANK} ./mpi_cuda_app

# Merge reports
nsys-ui report_*.nsys-rep
```

### **4.10.3 Continuous Integration**

```yaml
# .gitlab-ci.yml example
profile:
  stage: test
  script:
    - compute-sanitizer --tool memcheck ./program
    - ncu --set basic --export ci_metrics ./program
    - python3 check_performance_regression.py ci_metrics.ncu-rep
```

---

## ✅ **4.11 Summary**

**Key Takeaways:**
- Use Compute Sanitizer to catch errors early
- Profile system-wide with Nsight Systems
- Analyze kernels in detail with Nsight Compute
- Follow systematic optimization workflow
- Always measure before and after optimization

**Tool Quick Reference:**

| Task | Tool | Command |
|------|------|---------|
| Memory errors | Compute Sanitizer | `compute-sanitizer --tool memcheck` |
| Race conditions | Compute Sanitizer | `compute-sanitizer --tool racecheck` |
| System timeline | Nsight Systems | `nsys profile --stats=true` |
| Kernel metrics | Nsight Compute | `ncu --set full` |
| Register usage | NVCC | `nvcc -Xptxas -v` |

---

## 📚 **4.12 References**

### **Primary Documentation**
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Compute Sanitizer Guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

### **Additional Resources**
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

---

**Document Version:** 1.0
**Last Updated:** December 2024
**CUDA Toolkit Version:** 11.6+ (for Compute Sanitizer)

📄 **Next**: Advanced CUDA Optimization Techniques