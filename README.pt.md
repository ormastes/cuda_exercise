# Projeto CUDA Exercise

Um template de projeto de desenvolvimento CUDA com sistema de build CMake e integração VS Code.

**Versões de idioma:** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Русский](README.ru.md)

## Configurações Testadas

✅ **Testado com sucesso com:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Pré-requisitos

- CUDA Toolkit (12.0 ou posterior, testado com 13.0)
- CMake (3.24 ou posterior)
- Sistema de build Ninja
- Compilador Clang/LLVM (testado com Clang 20) ou GCC

## Construindo o Projeto

### Usando Linha de Comando

```bash
# Criar diretório de build
mkdir build
cd build

# Configurar com Ninja e Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Construir
ninja
```

### Usando VS Code

1. Abra a pasta do projeto no VS Code
2. Pressione `Ctrl+Shift+P` e execute "CMake: Select Configure Preset"
3. Escolha "default" para Debug ou "release" para build Release
4. Pressione `F7` para construir

## Estrutura do Projeto

```
.
├── CMakeLists.txt          # Configuração principal do CMake
├── CMakePresets.json       # Presets CMake para configuração fácil
├── demo/                   # Executável CUDA simples
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # Biblioteca CUDA com compilação separada
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # Configuração VS Code
    ├── settings.json      # Configurações CMake e build
    └── launch.json        # Configurações de debug
```

## Extensões VS Code para Desenvolvimento CUDA

### Extensões Obrigatórias

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Fornece IntelliSense, debugging e navegação de código
   - Inclui temas C/C++ e ferramentas CMake

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - Integração CMake com VS Code
   - Build, configuração e debug de projetos CMake

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - Suporte para debugging CUDA
   - Debug de kernel GPU
   - Integração CUDA-GDB

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Execute e faça debug de testes Catch2 do VS Code
   - Integração com explorador de testes
   - Indicadores visuais de status de teste

### Extensões Recomendadas

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - Destaque de sintaxe CUDA
   - Snippets para programação CUDA

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Snippets de código úteis para desenvolvimento C/C++

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Destaque de sintaxe aprimorado para C++ moderno

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternativa ao Microsoft C++ IntelliSense
   - Melhor suporte para recursos específicos do Clang
   - Completação de código e diagnósticos mais precisos

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - Integração do debugger LLDB
   - Melhor experiência de debug com código compilado pelo Clang

### Extensões Opcionais

10. **Error Lens** (`usernamehw.errorlens`)
    - Mostra erros e avisos inline

11. **GitLens** (`eamodio.gitlens`)
    - Integração Git aprimorada

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Verificação ortográfica para código e comentários

## Instalando Extensões VS Code

### Método 1: Scripts de Instalação Automatizada

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### Método 2: Através da Interface do VS Code
1. Abra o VS Code
2. Clique no ícone Extensões (Ctrl+Shift+X)
3. Procure por cada extensão pelo nome
4. Clique Instalar

### Método 3: Linha de Comando
```bash
# Instalar extensões obrigatórias
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Instalar extensões recomendadas
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Instalar extensões opcionais
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Debug

### Debug CUDA no VS Code

**Nota:** O debug CUDA atualmente é suportado apenas no Linux. Usuários Windows podem compilar e executar código CUDA mas não podem usar cuda-gdb para debug.

#### Linux
1. Defina breakpoints no seu código CUDA (arquivos .cu)
2. Pressione `F5` ou vá para Executar e Debug
3. Selecione a configuração "CUDA C++: Launch (cuda-gdb)"
4. O debugger irá parar nos breakpoints tanto no código host quanto device

### Configurações de Debug Disponíveis

- **CUDA C++: Launch (cuda-gdb)**: Debug código CUDA com cuda-gdb (somente Linux)
- Usa a seleção de target atual do CMake

## Arquivos de Configuração

### `.vscode/settings.json`
Configura CMake para usar Ninja e Clang:
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
Define presets de build para diferentes configurações:
- `default`: Build Debug com Clang e Ninja
- `release`: Build Release com otimizações

### `.vscode/launch.json`
Configurações de debug para aplicações CUDA com caminhos de debugger específicos da plataforma.

## Solução de Problemas

### Configuração Específica do Ambiente

Ao alterar seu ambiente de desenvolvimento ou versão CUDA, atualize os seguintes caminhos:

#### 1. Atualize debuggerPath em `.vscode/launch.json` (somente Linux)
Para usuários Linux, o caminho do debugger deve corresponder à sua instalação CUDA:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // Ou se instalado em local diferente:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

**Nota Windows:** O debug CUDA com cuda-gdb não é suportado no Windows. Usuários Windows podem compilar e executar aplicações CUDA mas devem usar métodos de debug alternativos como printf debugging ou NVIDIA Nsight Systems/Compute para profiling.

#### 2. Verifique o Caminho do CUDA Toolkit
Certifique-se de que o PATH do seu sistema inclui a versão CUDA correta:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### Aviso de Arquitetura CUDA
Se você ver "Cannot find valid GPU for '-arch=native'", isso significa que o CMake não consegue detectar a arquitetura da sua GPU. Você pode especificá-la manualmente em `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Para GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Para RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Para RTX 4090
# Ou para construir sem GPU (suporta múltiplas arquiteturas):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Compatibilidade de Versão Clang
CUDA pode não suportar oficialmente as versões mais recentes do Clang. O projeto usa a flag `-allow-unsupported-compiler` para contornar verificações de versão. Se encontrar problemas, considere usar GCC:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.