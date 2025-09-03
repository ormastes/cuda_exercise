# Progetto CUDA Exercise

Un template di progetto di sviluppo CUDA con sistema di build CMake e integrazione VS Code.

**Versioni linguistiche:** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## Configurazioni Testate

✅ **Testato con successo con:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Prerequisiti

- CUDA Toolkit (12.0 o successivo, testato con 13.0)
- CMake (3.24 o successivo)
- Sistema di build Ninja
- Compilatore Clang/LLVM (testato con Clang 20) o GCC

## Compilazione del Progetto

### Usando la Riga di Comando

```bash
# Crea la directory di build
mkdir build
cd build

# Configura con Ninja e Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Compila
ninja
```

### Usando VS Code

1. Apri la cartella del progetto in VS Code
2. Premi `Ctrl+Shift+P` ed esegui "CMake: Select Configure Preset"
3. Scegli "default" per Debug o "release" per build Release
4. Premi `F7` per compilare

## Struttura del Progetto

```
.
├── CMakeLists.txt          # Configurazione CMake principale
├── CMakePresets.json       # Preset CMake per configurazione facile
├── demo/                   # Eseguibile CUDA semplice
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # Libreria CUDA con compilazione separata
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # Configurazione VS Code
    ├── settings.json      # Impostazioni CMake e build
    └── launch.json        # Configurazioni di debug
```

## Estensioni VS Code per Sviluppo CUDA

### Estensioni Richieste

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Fornisce IntelliSense, debugging e navigazione del codice
   - Include temi C/C++ e strumenti CMake

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - Integrazione CMake con VS Code
   - Build, configurazione e debug di progetti CMake

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - Supporto per il debugging CUDA
   - Debug dei kernel GPU
   - Integrazione CUDA-GDB

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Esegui e fai debug di test Catch2 da VS Code
   - Integrazione con l'esploratore di test
   - Indicatori visivi dello stato dei test

### Estensioni Raccomandate

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - Evidenziazione della sintassi CUDA
   - Snippet per programmazione CUDA

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Snippet di codice utili per sviluppo C/C++

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Evidenziazione della sintassi migliorata per C++ moderno

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternativa a Microsoft C++ IntelliSense
   - Migliore supporto per funzionalità specifiche di Clang
   - Completamento del codice e diagnostica più accurati

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - Integrazione debugger LLDB
   - Migliore esperienza di debug con codice compilato con Clang

### Estensioni Opzionali

10. **Error Lens** (`usernamehw.errorlens`)
    - Mostra errori e avvertimenti inline

11. **GitLens** (`eamodio.gitlens`)
    - Integrazione Git avanzata

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Controllo ortografico per codice e commenti

## Installazione Estensioni VS Code

### Metodo 1: Script di Installazione Automatica

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### Metodo 2: Tramite Interfaccia VS Code
1. Apri VS Code
2. Clicca sull'icona Estensioni (Ctrl+Shift+X)
3. Cerca ogni estensione per nome
4. Clicca Installa

### Metodo 3: Riga di Comando
```bash
# Installa estensioni richieste
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Installa estensioni raccomandate
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Installa estensioni opzionali
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Debug

### Debug CUDA in VS Code

1. Imposta breakpoint nel tuo codice CUDA (file .cu)
2. Premi `F5` o vai a Esegui e Debug
3. Seleziona la configurazione "CUDA C++: Launch (cuda-gdb)"
4. Il debugger si fermerà ai breakpoint sia nel codice host che device

### Configurazioni di Debug Disponibili

- **CUDA C++: Launch (cuda-gdb)**: Debug codice CUDA con cuda-gdb
- Usa la selezione del target corrente di CMake
- Supporta piattaforme Linux e Windows

## File di Configurazione

### `.vscode/settings.json`
Configura CMake per usare Ninja e Clang:
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
Definisce preset di build per diverse configurazioni:
- `default`: Build Debug con Clang e Ninja
- `release`: Build Release con ottimizzazioni

### `.vscode/launch.json`
Configurazioni di debug per applicazioni CUDA con percorsi debugger specifici per piattaforma.

## Risoluzione Problemi

### Configurazione Specifica per Ambiente

Quando cambi il tuo ambiente di sviluppo o versione CUDA, aggiorna i seguenti percorsi:

#### 1. Aggiorna debuggerPath in `.vscode/launch.json`
Il percorso del debugger deve corrispondere alla versione e posizione della tua installazione CUDA:
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // Aggiorna a v13.0 o alla tua versione installata:
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Per Linux/WSL:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // O se installato in una posizione diversa:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. Verifica Percorso CUDA Toolkit
Assicurati che il PATH del tuo sistema includa la versione CUDA corretta:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### Avvertimento Architettura CUDA
Se vedi "Cannot find valid GPU for '-arch=native'", significa che CMake non può rilevare l'architettura della tua GPU. Puoi specificarla manualmente in `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Per GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Per RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Per RTX 4090
# O per compilare senza GPU (supporta architetture multiple):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Compatibilità Versione Clang
CUDA potrebbe non supportare ufficialmente le ultime versioni di Clang. Il progetto usa il flag `-allow-unsupported-compiler` per bypassare i controlli di versione. Se incontri problemi, considera l'uso di GCC invece:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Licenza

Questo progetto è licenziato sotto la Licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.