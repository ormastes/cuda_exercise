# Проект CUDA Exercise

Шаблон проекта разработки CUDA с системой сборки CMake и интеграцией VS Code.

**Языковые версии:** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md)

## Протестированные Конфигурации

✅ **Успешно протестировано с:**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Требования

- CUDA Toolkit (12.0 или новее, протестировано с 13.0)
- CMake (3.24 или новее)
- Система сборки Ninja
- Компилятор Clang/LLVM (протестировано с Clang 20) или GCC

## Сборка Проекта

### Используя Командную Строку

```bash
# Создать каталог сборки
mkdir build
cd build

# Настроить с Ninja и Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Собрать
ninja
```

### Используя VS Code

1. Откройте папку проекта в VS Code
2. Нажмите `Ctrl+Shift+P` и выполните "CMake: Select Configure Preset"
3. Выберите "default" для Debug или "release" для Release сборки
4. Нажмите `F7` для сборки

## Структура Проекта

```
.
├── CMakeLists.txt          # Основная конфигурация CMake
├── CMakePresets.json       # Пресеты CMake для легкой настройки
├── demo/                   # Простой исполняемый файл CUDA
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # Библиотека CUDA с раздельной компиляцией
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # Конфигурация VS Code
    ├── settings.json      # Настройки CMake и сборки
    └── launch.json        # Конфигурации отладки
```

## Расширения VS Code для Разработки CUDA

### Обязательные Расширения

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Обеспечивает IntelliSense, отладку и навигацию по коду
   - Включает темы C/C++ и инструменты CMake

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - Интеграция CMake с VS Code
   - Сборка, настройка и отладка проектов CMake

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - Поддержка отладки CUDA
   - Отладка ядер GPU
   - Интеграция CUDA-GDB

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Запуск и отладка тестов Catch2 из VS Code
   - Интеграция с проводником тестов
   - Визуальные индикаторы состояния тестов

### Рекомендуемые Расширения

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - Подсветка синтаксиса CUDA
   - Сниппеты для программирования CUDA

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Полезные сниппеты кода для разработки C/C++

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Улучшенная подсветка синтаксиса для современного C++

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Альтернатива Microsoft C++ IntelliSense
   - Лучшая поддержка специфичных для Clang функций
   - Более точное автодополнение кода и диагностика

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - Интеграция отладчика LLDB
   - Лучший опыт отладки с кодом, скомпилированным Clang

### Дополнительные Расширения

10. **Error Lens** (`usernamehw.errorlens`)
    - Показывает ошибки и предупреждения встроенно

11. **GitLens** (`eamodio.gitlens`)
    - Расширенная интеграция Git

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Проверка орфографии для кода и комментариев

## Установка Расширений VS Code

### Метод 1: Автоматизированные Скрипты Установки

**Linux/macOS:**
```bash
./install-vscode-extensions.sh
```

**Windows:**
```batch
install-vscode-extensions.bat
```

### Метод 2: Через Интерфейс VS Code
1. Откройте VS Code
2. Щелкните на иконку Расширения (Ctrl+Shift+X)
3. Найдите каждое расширение по имени
4. Нажмите Установить

### Метод 3: Командная Строка
```bash
# Установить обязательные расширения
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Установить рекомендуемые расширения
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Установить дополнительные расширения
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Отладка

### Отладка CUDA в VS Code

1. Установите точки прерывания в вашем коде CUDA (файлы .cu)
2. Нажмите `F5` или перейдите в Запуск и Отладка
3. Выберите конфигурацию "CUDA C++: Launch (cuda-gdb)"
4. Отладчик остановится на точках прерывания как в коде хоста, так и устройства

### Доступные Конфигурации Отладки

- **CUDA C++: Launch (cuda-gdb)**: Отладка кода CUDA с cuda-gdb
- Использует текущий выбор целевого объекта CMake
- Поддерживает платформы Linux и Windows

## Файлы Конфигурации

### `.vscode/settings.json`
Настраивает CMake для использования Ninja и Clang:
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
Определяет пресеты сборки для различных конфигураций:
- `default`: Debug сборка с Clang и Ninja
- `release`: Release сборка с оптимизациями

### `.vscode/launch.json`
Конфигурации отладки для приложений CUDA с путями отладчика, специфичными для платформы.

## Устранение Проблем

### Конфигурация Специфичная для Среды

При изменении среды разработки или версии CUDA, обновите следующие пути:

#### 1. Обновите debuggerPath в `.vscode/launch.json`
Путь отладчика должен соответствовать версии и расположению вашей установки CUDA:
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // Обновите до v13.0 или вашей установленной версии:
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Для Linux/WSL:
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // Или если установлено в другом месте:
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. Проверьте Путь CUDA Toolkit
Убедитесь, что PATH вашей системы включает правильную версию CUDA:
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux: `/usr/local/cuda-13.0/bin`

### Предупреждение Архитектуры CUDA
Если вы видите "Cannot find valid GPU for '-arch=native'", это означает, что CMake не может определить архитектуру вашего GPU. Вы можете указать её вручную в `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Для GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Для RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Для RTX 4090
# Или для сборки без GPU (поддерживает множественные архитектуры):
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Совместимость Версий Clang
CUDA может официально не поддерживать последние версии Clang. Проект использует флаг `-allow-unsupported-compiler` для обхода проверок версий. Если возникают проблемы, рассмотрите использование GCC вместо этого:
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Лицензия

Этот проект лицензирован под Лицензией MIT - см. файл [LICENSE](LICENSE) для подробностей.