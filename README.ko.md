# CUDA 연습 프로젝트

CMake 빌드 시스템과 VS Code 통합을 갖춘 CUDA 개발 프로젝트 템플릿입니다.

**언어 버전:** [English](README.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## 테스트된 구성

✅ **성공적으로 테스트됨:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## 필수 요구 사항

- CUDA Toolkit (12.0 이상, 13.0으로 테스트됨)
- CMake (3.24 이상)
- Ninja 빌드 시스템
- Clang/LLVM 컴파일러 (Clang 20으로 테스트됨) 또는 GCC

## 프로젝트 빌드하기

### 명령줄 사용

```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# Ninja와 Clang으로 구성
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# 빌드
ninja
```

### VS Code 사용

1. VS Code에서 프로젝트 폴더 열기
2. `Ctrl+Shift+P`를 누르고 "CMake: Select Configure Preset" 실행
3. Debug 빌드는 "default", Release 빌드는 "release" 선택
4. `F7`을 눌러 빌드

## 프로젝트 구조

```
.
├── CMakeLists.txt          # 메인 CMake 구성
├── CMakePresets.json       # 쉬운 구성을 위한 CMake 프리셋
├── demo/                   # 간단한 CUDA 실행 파일
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # 분리 컴파일이 있는 CUDA 라이브러리
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # VS Code 구성
    ├── settings.json      # CMake 및 빌드 설정
    └── launch.json        # 디버그 구성
```

## CUDA 개발을 위한 VS Code 확장

### 필수 확장

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - IntelliSense, 디버깅 및 코드 탐색 제공
   - C/C++ 테마 및 CMake Tools 포함

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - VS Code와 CMake 통합
   - CMake 프로젝트 빌드, 구성 및 디버그

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - CUDA 디버깅 지원
   - GPU 커널 디버깅
   - CUDA-GDB 통합

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - VS Code에서 Catch2 테스트 실행 및 디버그
   - 테스트 탐색기 통합
   - 시각적 테스트 상태 표시기

### 권장 확장

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - CUDA 구문 강조
   - CUDA 프로그래밍용 스니펫

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - C/C++ 개발용 유용한 코드 스니펫

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - 현대 C++용 향상된 구문 강조

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Microsoft C++ IntelliSense 대안
   - Clang 특정 기능에 대한 더 나은 지원
   - 더 정확한 코드 완성 및 진단

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - LLDB 디버거 통합
   - Clang으로 컴파일된 코드에 대한 더 나은 디버깅 경험

### 선택적 확장

10. **Error Lens** (`usernamehw.errorlens`)
    - 오류 및 경고를 인라인으로 표시

11. **GitLens** (`eamodio.gitlens`)
    - 향상된 Git 통합

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - 코드 및 주석에 대한 맞춤법 검사

## VS Code 확장 설치

### 방법 1: 자동 설치 스크립트

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### 방법 2: VS Code UI를 통해

1. VS Code 열기
2. 확장 아이콘 클릭 (Ctrl+Shift+X)
3. 이름으로 각 확장 검색
4. 설치 클릭

### 방법 3: 명령줄

```bash
# 필수 확장 설치
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# 권장 확장 설치
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# 선택적 확장 설치
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## 디버깅

### VS Code에서 CUDA 디버깅

1. CUDA 코드(.cu 파일)에 중단점 설정
2. `F5`를 누르거나 실행 및 디버그로 이동
3. "CUDA C++: Launch (cuda-gdb)" 구성 선택
4. 디버거는 호스트 및 장치 코드 모두에서 중단점에서 멈춥니다

### 사용 가능한 디버그 구성

- **CUDA C++: Launch (cuda-gdb)**: cuda-gdb로 CUDA 코드 디버그
- CMake의 현재 대상 선택 사용
- Linux 및 Windows 플랫폼 모두 지원

## 구성 파일

### `.vscode/settings.json`
Ninja와 Clang을 사용하도록 CMake 구성:
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
다양한 구성을 위한 빌드 프리셋 정의:
- `default`: Clang과 Ninja를 사용한 Debug 빌드
- `default-msvc`: Visual Studio 생성기를 사용한 MSVC 빌드
- `release`: 최적화가 포함된 Release 빌드

### `.vscode/launch.json`
플랫폼별 디버거 경로가 있는 CUDA 애플리케이션용 디버그 구성.

## 문제 해결

### 환경별 구성

개발 환경이나 CUDA 버전을 변경할 때 다음 경로를 업데이트하세요:

#### 1. `.vscode/launch.json`에서 debuggerPath 업데이트
디버거 경로는 CUDA 설치 버전 및 위치와 일치해야 합니다:
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // v13.0 또는 설치된 버전으로 업데이트:
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Linux/WSL의 경우:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // 또는 다른 위치에 설치된 경우:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. CUDA Toolkit 경로 확인
시스템 PATH에 올바른 CUDA 버전이 포함되어 있는지 확인하세요:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### CUDA 아키텍처 경고
"Cannot find valid GPU for '-arch=native'" 오류가 표시되면 CMake가 GPU 아키텍처를 감지할 수 없음을 의미합니다. `CMakeLists.txt`에서 수동으로 지정할 수 있습니다:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # GTX 1660 Ti, RTX 2060-2080용
set(CMAKE_CUDA_ARCHITECTURES "86")  # RTX 3060-3090용
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4090용
# 또는 GPU 없이 빌드하기 위해 (여러 아키텍처 지원):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Clang 버전 호환성
CUDA는 최신 Clang 버전을 공식적으로 지원하지 않을 수 있습니다. 프로젝트는 버전 확인을 우회하기 위해 `-allow-unsupported-compiler` 플래그를 사용합니다. 문제가 발생하면 대신 GCC 사용을 고려하세요:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## 라이선스

이 프로젝트는 MIT 라이선스로 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.