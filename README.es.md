# Proyecto de Ejercicio CUDA

Una plantilla de proyecto de desarrollo CUDA con sistema de construcción CMake e integración con VS Code.

**Versiones de idioma:** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## Configuraciones Probadas

✅ **Probado exitosamente con:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Prerrequisitos

- CUDA Toolkit (12.0 o posterior, probado con 13.0)
- CMake (3.24 o posterior)
- Sistema de construcción Ninja
- Compilador Clang/LLVM (probado con Clang 20) o GCC

## Construcción del Proyecto

### Usando Línea de Comandos

```bash
# Crear directorio de construcción
mkdir build
cd build

# Configurar con Ninja y Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Construir
ninja
```

### Usando VS Code

1. Abrir la carpeta del proyecto en VS Code
2. Presionar `Ctrl+Shift+P` y ejecutar "CMake: Select Configure Preset"
3. Elegir "default" para Debug o "release" para Release build
4. Presionar `F7` para construir

## Estructura del Proyecto

```
.
├── CMakeLists.txt          # Configuración principal de CMake
├── CMakePresets.json       # Presets de CMake para configuración fácil
├── demo/                   # Ejecutable CUDA simple
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # Biblioteca CUDA con compilación separada
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # Configuración de VS Code
    ├── settings.json      # Configuraciones de CMake y construcción
    └── launch.json        # Configuraciones de depuración
```

## Extensiones de VS Code para Desarrollo CUDA

### Extensiones Requeridas

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Proporciona IntelliSense, depuración y navegación de código
   - Incluye temas de C/C++ y CMake Tools

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - Integración de CMake con VS Code
   - Construcción, configuración y depuración de proyectos CMake

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - Soporte de depuración CUDA
   - Depuración de kernels de GPU
   - Integración CUDA-GDB

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Ejecutar y depurar pruebas Catch2 desde VS Code
   - Integración del explorador de pruebas
   - Indicadores visuales de estado de pruebas

### Extensiones Recomendadas

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - Resaltado de sintaxis CUDA
   - Fragmentos de código para programación CUDA

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Fragmentos de código útiles para desarrollo C/C++

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Resaltado de sintaxis mejorado para C++ moderno

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternativa a Microsoft C++ IntelliSense
   - Mejor soporte para características específicas de Clang
   - Autocompletado y diagnósticos más precisos

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - Integración del depurador LLDB
   - Mejor experiencia de depuración con código compilado por Clang

### Extensiones Opcionales

10. **Error Lens** (`usernamehw.errorlens`)
    - Muestra errores y advertencias en línea

11. **GitLens** (`eamodio.gitlens`)
    - Integración Git mejorada

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Verificación ortográfica para código y comentarios

## Instalación de Extensiones de VS Code

### Método 1: Scripts de Instalación Automatizada

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### Método 2: A través de la UI de VS Code
1. Abrir VS Code
2. Hacer clic en el icono de Extensiones (Ctrl+Shift+X)
3. Buscar cada extensión por nombre
4. Hacer clic en Instalar

### Método 3: Línea de Comandos
```bash
# Instalar extensiones requeridas
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Instalar extensiones recomendadas
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Instalar extensiones opcionales
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Depuración

### Depuración CUDA en VS Code

**Nota:** La depuración CUDA actualmente solo está soportada en Linux. Los usuarios de Windows pueden compilar y ejecutar código CUDA pero no pueden usar cuda-gdb para depuración.

#### Linux
1. Establecer puntos de interrupción en su código CUDA (archivos .cu)
2. Presionar `F5` o ir a Ejecutar y Depurar
3. Seleccionar la configuración "CUDA C++: Launch (cuda-gdb)"
4. El depurador se detendrá en los puntos de interrupción tanto en código host como dispositivo

### Configuraciones de Depuración Disponibles

- **CUDA C++: Launch (cuda-gdb)**: Depurar código CUDA con cuda-gdb (solo Linux)
- Usa la selección de target actual de CMake

## Archivos de Configuración

### `.vscode/settings.json`
Configura CMake para usar Ninja y Clang:
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
Define presets de construcción para diferentes configuraciones:
- `default`: Construcción Debug con Clang y Ninja
- `release`: Construcción Release con optimizaciones

### `.vscode/launch.json`
Configuraciones de depuración para aplicaciones CUDA con rutas de depurador específicas de plataforma.

## Solución de Problemas

### Configuración Específica del Entorno

Al cambiar su entorno de desarrollo o versión de CUDA, actualice las siguientes rutas:

#### 1. Actualizar debuggerPath en `.vscode/launch.json` (solo Linux)
Para usuarios de Linux, la ruta del depurador debe coincidir con su instalación CUDA:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // O si está instalado en una ubicación diferente:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

**Nota Windows:** La depuración CUDA con cuda-gdb no está soportada en Windows. Los usuarios de Windows pueden compilar y ejecutar aplicaciones CUDA pero deben usar métodos de depuración alternativos como depuración con printf o NVIDIA Nsight Systems/Compute para perfilado.

#### 2. Verificar Ruta de CUDA Toolkit
Asegúrese de que su PATH del sistema incluya la versión CUDA correcta:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### Advertencia de Arquitectura CUDA
Si ve "Cannot find valid GPU for '-arch=native'", esto significa que CMake no puede detectar su arquitectura de GPU. Puede especificarla manualmente en `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Para GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Para RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Para RTX 4090
# O para construcción sin GPU (soporta múltiples arquitecturas):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Compatibilidad de Versión Clang
CUDA puede no soportar oficialmente las últimas versiones de Clang. El proyecto usa la bandera `-allow-unsupported-compiler` para omitir las verificaciones de versión. Si encuentra problemas, considere usar GCC en su lugar:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.