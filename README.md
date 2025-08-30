# CUDA Exercise Project

A CUDA development project template with CMake build system and VS Code integration.

## Prerequisites

- CUDA Toolkit (12.0 or later)
- CMake (3.24 or later)
- Ninja build system
- Clang/LLVM compiler (or GCC)

## Building the Project

### Using Command Line

```bash
# Create build directory
mkdir build
cd build

# Configure with Ninja and Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Build
ninja
```

### Using VS Code

1. Open the project folder in VS Code
2. Press `Ctrl+Shift+P` and run "CMake: Select Configure Preset"
3. Choose "default" for Debug or "release" for Release build
4. Press `F7` to build

## Project Structure

```
.
├── CMakeLists.txt          # Main CMake configuration
├── CMakePresets.json       # CMake presets for easy configuration
├── demo/                   # Simple CUDA executable
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # CUDA library with separate compilation
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # VS Code configuration
    ├── settings.json      # CMake and build settings
    └── launch.json        # Debug configurations
```

## VS Code Extensions for CUDA Development

### Required Extensions

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Provides IntelliSense, debugging, and code browsing
   - Includes C/C++ themes and CMake Tools

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - CMake integration with VS Code
   - Build, configure, and debug CMake projects
   - Already included in C/C++ Extension Pack

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - CUDA debugging support
   - GPU kernel debugging
   - CUDA-GDB integration

### Recommended Extensions

4. **CMake Language Support** (`twxs.cmake`)
   - Syntax highlighting for CMake files
   - Auto-completion for CMake commands

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - CUDA syntax highlighting
   - Snippets for CUDA programming

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Useful code snippets for C/C++ development

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Enhanced syntax highlighting for modern C++

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternative to Microsoft C++ IntelliSense
   - Better support for Clang-specific features
   - More accurate code completion and diagnostics

### Optional Extensions

9. **Error Lens** (`usernamehw.errorlens`)
   - Shows errors and warnings inline

10. **GitLens** (`eamodio.gitlens`)
    - Enhanced Git integration

11. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Spell checking for code and comments

## Installing VS Code Extensions

### Method 1: Through VS Code UI
1. Open VS Code
2. Click on Extensions icon (Ctrl+Shift+X)
3. Search for each extension by name
4. Click Install

### Method 2: Command Line
```bash
# Install required extensions
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition

# Install recommended extensions
code --install-extension twxs.cmake
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd

# Install optional extensions
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Debugging

### CUDA Debugging in VS Code

1. Set breakpoints in your CUDA code (.cu files)
2. Press `F5` or go to Run and Debug
3. Select "CUDA C++: Launch (cuda-gdb)" configuration
4. The debugger will stop at breakpoints in both host and device code

### Available Debug Configurations

- **CUDA C++: Launch (cuda-gdb)**: Debug CUDA code with cuda-gdb
- Uses CMake's current target selection
- Supports both Linux and Windows platforms

## Configuration Files

### `.vscode/settings.json`
Configures CMake to use Ninja and Clang:
```json
{
    "cmake.generator": "Ninja",
    "cmake.configureArgs": [
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE"
    ]
}
```

### `CMakePresets.json`
Defines build presets for different configurations:
- `default`: Debug build with Clang and Ninja
- `release`: Release build with optimizations

### `.vscode/launch.json`
Debug configurations for CUDA applications with platform-specific debugger paths.

## Troubleshooting

### CUDA Architecture Warning
If you see "Cannot find valid GPU for '-arch=native'", this means CMake cannot detect your GPU architecture. You can specify it manually in `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # For GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # For RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # For RTX 4090
```

### Clang Version Compatibility
CUDA may not officially support the latest Clang versions. The project uses `-allow-unsupported-compiler` flag to bypass version checks. If you encounter issues, consider using GCC instead:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.