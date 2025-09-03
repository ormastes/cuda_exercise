# CUDA Exercise Project

Een CUDA ontwikkelingsproject template met CMake build systeem en VS Code integratie.

**Taalversies:** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## Geteste Configuraties

✅ **Succesvol getest met:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Vereisten

- CUDA Toolkit (12.0 of later, getest met 13.0)
- CMake (3.24 of later)
- Ninja build systeem
- Clang/LLVM compiler (getest met Clang 20) of GCC

## Het Project Bouwen

### Met Commandoregel

```bash
# Maak build directory aan
mkdir build
cd build

# Configureer met Ninja en Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Bouw
ninja
```

### Met VS Code

1. Open de projectmap in VS Code
2. Druk `Ctrl+Shift+P` en voer "CMake: Select Configure Preset" uit
3. Kies "default" voor Debug of "release" voor Release build
4. Druk `F7` om te bouwen

## Projectstructuur

```
.
├── CMakeLists.txt          # Hoofd CMake configuratie
├── CMakePresets.json       # CMake presets voor eenvoudige configuratie
├── demo/                   # Eenvoudige CUDA executable
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # CUDA bibliotheek met gescheiden compilatie
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # VS Code configuratie
    ├── settings.json      # CMake en build instellingen
    └── launch.json        # Debug configuraties
```

## VS Code Extensies voor CUDA Ontwikkeling

### Vereiste Extensies

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Biedt IntelliSense, debugging en code browsing
   - Bevat C/C++ themes en CMake Tools

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - CMake integratie met VS Code
   - Build, configureer en debug CMake projecten

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - CUDA debugging ondersteuning
   - GPU kernel debugging
   - CUDA-GDB integratie

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Voer Catch2 tests uit en debug ze vanuit VS Code
   - Test explorer integratie
   - Visuele test status indicatoren

### Aanbevolen Extensies

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - CUDA syntax highlighting
   - Snippets voor CUDA programmering

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Nuttige code snippets voor C/C++ ontwikkeling

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Verbeterde syntax highlighting voor modern C++

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternatief voor Microsoft C++ IntelliSense
   - Betere ondersteuning voor Clang-specifieke functies
   - Nauwkeurigere code aanvulling en diagnostiek

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - LLDB debugger integratie
   - Betere debugging ervaring met Clang-gecompileerde code

### Optionele Extensies

10. **Error Lens** (`usernamehw.errorlens`)
    - Toont fouten en waarschuwingen inline

11. **GitLens** (`eamodio.gitlens`)
    - Verbeterde Git integratie

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Spellingcontrole voor code en opmerkingen

## VS Code Extensies Installeren

### Methode 1: Geautomatiseerde Installatie Scripts

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### Methode 2: Via VS Code UI
1. Open VS Code
2. Klik op Extensies icoon (Ctrl+Shift+X)
3. Zoek naar elke extensie op naam
4. Klik Installeer

### Methode 3: Commandoregel
```bash
# Installeer vereiste extensies
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Installeer aanbevolen extensies
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Installeer optionele extensies
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Debuggen

### CUDA Debuggen in VS Code

1. Zet breakpoints in je CUDA code (.cu bestanden)
2. Druk `F5` of ga naar Run and Debug
3. Selecteer "CUDA C++: Launch (cuda-gdb)" configuratie
4. De debugger stopt bij breakpoints in zowel host als device code

### Beschikbare Debug Configuraties

- **CUDA C++: Launch (cuda-gdb)**: Debug CUDA code met cuda-gdb
- Gebruikt CMake's huidige target selectie
- Ondersteunt zowel Linux als Windows platforms

## Configuratiebestanden

### `.vscode/settings.json`
Configureert CMake om Ninja en Clang te gebruiken:
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
Definieert build presets voor verschillende configuraties:
- `default`: Debug build met Clang en Ninja
- `release`: Release build met optimalisaties

### `.vscode/launch.json`
Debug configuraties voor CUDA applicaties met platform-specifieke debugger paden.

## Probleemoplossing

### Omgeving-Specifieke Configuratie

Bij het wijzigen van je ontwikkelomgeving of CUDA versie, werk de volgende paden bij:

#### 1. Update debuggerPath in `.vscode/launch.json`
Het debugger pad moet overeenkomen met je CUDA installatie versie en locatie:
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // Update naar v13.0 of je geïnstalleerde versie:
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Voor Linux/WSL:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // Of indien geïnstalleerd op een andere locatie:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. Verifieer CUDA Toolkit Pad
Zorg ervoor dat je systeem PATH de juiste CUDA versie bevat:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### CUDA Architectuur Waarschuwing
Als je "Cannot find valid GPU for '-arch=native'" ziet, betekent dit dat CMake je GPU architectuur niet kan detecteren. Je kunt het handmatig specificeren in `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Voor GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Voor RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Voor RTX 4090
# Of voor bouwen zonder GPU (ondersteunt meerdere architecturen):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Clang Versie Compatibiliteit
CUDA ondersteunt mogelijk niet officieel de nieuwste Clang versies. Het project gebruikt `-allow-unsupported-compiler` vlag om versie controles te omzeilen. Als je problemen ondervindt, overweeg dan GCC te gebruiken:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Licentie

Dit project is gelicenseerd onder de MIT Licentie - zie het [LICENSE](LICENSE) bestand voor details.