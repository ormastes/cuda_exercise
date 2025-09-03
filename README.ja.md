# CUDA演習プロジェクト

CMakeビルドシステムとVS Code統合を備えたCUDA開発プロジェクトテンプレートです。

**言語版:** [English](README.md) | [한국어](README.ko.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## テスト済み構成

✅ **次の環境でテスト済み:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## 前提条件

- CUDA Toolkit（12.0以降、13.0でテスト済み）
- CMake（3.24以降）
- Ninjaビルドシステム
- Clang/LLVMコンパイラ（Clang 20でテスト済み）またはGCC

## プロジェクトのビルド

### コマンドラインを使用

```bash
# ビルドディレクトリを作成
mkdir build
cd build

# NinjaとClangで構成
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# ビルド
ninja
```

### VS Codeを使用

1. VS Codeでプロジェクトフォルダーを開く
2. `Ctrl+Shift+P`を押して「CMake: Select Configure Preset」を実行
3. デバッグビルドの場合は「default」、リリースビルドの場合は「release」を選択
4. `F7`を押してビルド

## プロジェクト構造

```
.
├── CMakeLists.txt          # メインCMake構成
├── CMakePresets.json       # 簡単な構成のためのCMakeプリセット
├── demo/                   # シンプルなCUDA実行可能ファイル
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # 個別コンパイルを使用したCUDAライブラリ
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # VS Code構成
    ├── settings.json      # CMakeとビルド設定
    └── launch.json        # デバッグ構成
```

## CUDA開発用VS Code拡張機能

### 必須拡張機能

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - IntelliSense、デバッグ、コードブラウジングを提供
   - C/C++テーマとCMake Toolsを含む

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - VS CodeとのCMake統合
   - CMakeプロジェクトのビルド、構成、デバッグ

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - CUDAデバッグサポート
   - GPUカーネルデバッグ
   - CUDA-GDB統合

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - VS CodeからCatch2テストの実行とデバッグ
   - テストエクスプローラー統合
   - 視覚的なテストステータス表示

### 推奨拡張機能

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - CUDAシンタックスハイライト
   - CUDAプログラミング用スニペット

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - C/C++開発用の便利なコードスニペット

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - モダンC++用の拡張シンタックスハイライト

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Microsoft C++ IntelliSenseの代替
   - Clang固有機能のより良いサポート
   - より正確なコード補完と診断

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - LLDBデバッガー統合
   - Clangコンパイルコードのより良いデバッグ体験

### オプション拡張機能

10. **Error Lens** (`usernamehw.errorlens`)
    - エラーと警告をインラインで表示

11. **GitLens** (`eamodio.gitlens`)
    - 拡張Git統合

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - コードとコメントのスペルチェック

## VS Code拡張機能のインストール

### 方法1: 自動インストールスクリプト

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### 方法2: VS Code UIを通じて
1. VS Codeを開く
2. 拡張機能アイコンをクリック（Ctrl+Shift+X）
3. 各拡張機能を名前で検索
4. インストールをクリック

### 方法3: コマンドライン
```bash
# 必須拡張機能をインストール
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# 推奨拡張機能をインストール
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# オプション拡張機能をインストール
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## デバッグ

### VS CodeでのCUDAデバッグ

1. CUDAコード（.cuファイル）にブレークポイントを設定
2. `F5`を押すか、実行とデバッグに移動
3. 「CUDA C++: Launch (cuda-gdb)」構成を選択
4. デバッガーはホストとデバイスコードの両方のブレークポイントで停止

### 利用可能なデバッグ構成

- **CUDA C++: Launch (cuda-gdb)**: cuda-gdbでCUDAコードをデバッグ
- CMakeの現在のターゲット選択を使用
- LinuxとWindowsプラットフォームの両方をサポート

## 構成ファイル

### `.vscode/settings.json`
NinjaとClangを使用するようCMakeを構成:
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
異なる構成のビルドプリセットを定義:
- `default`: ClangとNinjaを使用したデバッグビルド
- `release`: 最適化を有効にしたリリースビルド

### `.vscode/launch.json`
プラットフォーム固有のデバッガーパスを使用したCUDAアプリケーションのデバッグ構成。

## トラブルシューティング

### 環境固有の構成

開発環境やCUDAバージョンを変更する際は、以下のパスを更新してください:

#### 1. `.vscode/launch.json`のdebuggerPathを更新

デバッガーパスはCUDAインストールバージョンと場所に一致する必要があります:
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // v13.0またはインストールしたバージョンに更新:
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Linux/WSLの場合:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // または異なる場所にインストールされている場合:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. CUDA Toolkitパスを確認

システムPATHに正しいCUDAバージョンが含まれていることを確認:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### CUDAアーキテクチャ警告
「'-arch=native'の有効なGPUが見つかりません」というメッセージが表示される場合、CMakeがGPUアーキテクチャを検出できないことを意味します。`CMakeLists.txt`で手動で指定できます:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # GTX 1660 Ti、RTX 2060-2080用
set(CMAKE_CUDA_ARCHITECTURES "86")  # RTX 3060-3090用
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4090用
# またはGPUなしでのビルド（複数アーキテクチャをサポート）:
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Clangバージョン互換性
CUDAは最新のClangバージョンを公式にサポートしていない可能性があります。このプロジェクトは`-allow-unsupported-compiler`フラグを使用してバージョンチェックをバイパスします。問題が発生した場合は、代わりにGCCの使用を検討してください:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。