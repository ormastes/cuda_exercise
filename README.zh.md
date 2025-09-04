# CUDA 练习项目

一个具有 CMake 构建系统和 VS Code 集成的 CUDA 开发项目模板。

**语言版本：** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## 测试配置

✅ **成功测试环境：**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## 先决条件

- CUDA Toolkit（12.0 或更新版本，已在 13.0 上测试）
- CMake（3.24 或更新版本）
- Ninja 构建系统
- Clang/LLVM 编译器（已在 Clang 20 上测试）或 GCC

## 构建项目

### 使用命令行

```bash
# 创建构建目录
mkdir build
cd build

# 使用 Ninja 和 Clang 配置
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# 构建
ninja
```

### 使用 VS Code

1. 在 VS Code 中打开项目文件夹
2. 按 `Ctrl+Shift+P` 并运行 "CMake: Select Configure Preset"
3. 选择 "default" 进行调试构建或 "release" 进行发布构建
4. 按 `F7` 进行构建

## 项目结构

```
.
├── CMakeLists.txt          # 主 CMake 配置
├── CMakePresets.json       # 便于配置的 CMake 预设
├── demo/                   # 简单的 CUDA 可执行文件
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # 具有单独编译的 CUDA 库
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # VS Code 配置
    ├── settings.json      # CMake 和构建设置
    └── launch.json        # 调试配置
```

## 用于 CUDA 开发的 VS Code 扩展

### 必需扩展

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - 提供 IntelliSense、调试和代码浏览
   - 包括 C/C++ 主题和 CMake Tools

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - CMake 与 VS Code 的集成
   - 构建、配置和调试 CMake 项目

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - CUDA 调试支持
   - GPU 内核调试
   - CUDA-GDB 集成

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - 从 VS Code 运行和调试 Catch2 测试
   - 测试资源管理器集成
   - 可视化测试状态指示器

### 推荐扩展

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - CUDA 语法高亮
   - CUDA 编程代码片段

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - C/C++ 开发的有用代码片段

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - 现代 C++ 的增强语法高亮

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Microsoft C++ IntelliSense 的替代方案
   - 更好地支持 Clang 特定功能
   - 更准确的代码补全和诊断

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - LLDB 调试器集成
   - 使用 Clang 编译代码的更好调试体验

### 可选扩展

10. **Error Lens** (`usernamehw.errorlens`)
    - 内联显示错误和警告

11. **GitLens** (`eamodio.gitlens`)
    - 增强的 Git 集成

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - 代码和注释的拼写检查

## 安装 VS Code 扩展

### 方法 1：自动安装脚本

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### 方法 2：通过 VS Code UI
1. 打开 VS Code
2. 点击扩展图标（Ctrl+Shift+X）
3. 按名称搜索每个扩展
4. 点击安装

### 方法 3：命令行
```bash
# 安装必需扩展
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# 安装推荐扩展
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# 安装可选扩展
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## 调试

### VS Code 中的 CUDA 调试

**注意：**CUDA 调试目前仅在 Linux 上支持。Windows 用户可以编译和运行 CUDA 代码，但不能使用 cuda-gdb 进行调试。

#### Linux
1. 在您的 CUDA 代码（.cu 文件）中设置断点
2. 按 `F5` 或转到运行和调试
3. 选择 "CUDA C++: Launch (cuda-gdb)" 配置
4. 调试器将在主机和设备代码的断点处停止

### 可用的调试配置

- **CUDA C++: Launch (cuda-gdb)**：使用 cuda-gdb 调试 CUDA 代码（仅 Linux）
- 使用 CMake 的当前目标选择

## 配置文件

### `.vscode/settings.json`
配置 CMake 使用 Ninja 和 Clang：
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
定义不同配置的构建预设：
- `default`：使用 Clang 和 Ninja 的调试构建
- `release`：带优化的发布构建

### `.vscode/launch.json`
具有平台特定调试器路径的 CUDA 应用程序调试配置。

## 故障排除

### 环境特定配置

更改开发环境或 CUDA 版本时，请更新以下路径：

#### 1. 更新 `.vscode/launch.json` 中的 debuggerPath（仅 Linux）
对于 Linux 用户，调试器路径必须与您的 CUDA 安装相匹配：
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // 或如果安装在不同位置：
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

**Windows 注意：**Windows 上不支持使用 cuda-gdb 进行 CUDA 调试。Windows 用户可以编译和运行 CUDA 应用程序，但应使用替代调试方法，如 printf 调试或使用 NVIDIA Nsight Systems/Compute 进行分析。

#### 2. 验证 CUDA Toolkit 路径
确保您的系统 PATH 包含正确的 CUDA 版本：
- Windows：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux：`/usr/local/cuda-13.0/bin`

### CUDA 架构警告
如果您看到 "Cannot find valid GPU for '-arch=native'"，这意味着 CMake 无法检测您的 GPU 架构。您可以在 `CMakeLists.txt` 中手动指定：
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # 用于 GTX 1660 Ti、RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # 用于 RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # 用于 RTX 4090
# 或用于无 GPU 构建（支持多种架构）：
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Clang 版本兼容性
CUDA 可能不正式支持最新的 Clang 版本。该项目使用 `-allow-unsupported-compiler` 标志来绕过版本检查。如果遇到问题，请考虑使用 GCC：
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## 许可证

该项目在 MIT 许可证下授权 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。