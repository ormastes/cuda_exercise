# Projet d'Exercice CUDA

Un modèle de projet de développement CUDA avec système de construction CMake et intégration VS Code.

**Versions linguistiques :** [English](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [Deutsch](README.de.md) | [中文](README.zh.md) | [Español](README.es.md) | [Italiano](README.it.md) | [Nederlands](README.nl.md) | [Português](README.pt.md) | [Русский](README.ru.md)

## Configurations Testées

✅ **Testé avec succès avec :**
- Ubuntu 24.04 LTS
- Clang 20
- CUDA Toolkit 13.0
- CMake 3.28+
- Ninja 1.11+

## Prérequis

- CUDA Toolkit (12.0 ou plus récent, testé avec 13.0)
- CMake (3.24 ou plus récent)
- Système de construction Ninja
- Compilateur Clang/LLVM (testé avec Clang 20) ou GCC

## Construction du Projet

### Utilisation de la Ligne de Commande

```bash
# Créer le répertoire de construction
mkdir build
cd build

# Configurer avec Ninja et Clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug ..

# Construire
ninja
```

### Utilisation de VS Code

1. Ouvrir le dossier du projet dans VS Code
2. Appuyez sur `Ctrl+Shift+P` et exécutez "CMake: Select Configure Preset"
3. Choisissez "default" pour Debug ou "release" pour Release build
4. Appuyez sur `F7` pour construire

## Structure du Projet

```
.
├── CMakeLists.txt          # Configuration principale CMake
├── CMakePresets.json       # Préréglages CMake pour une configuration facile
├── demo/                   # Exécutable CUDA simple
│   ├── CMakeLists.txt
│   └── src.cu
├── demo_lib/              # Bibliothèque CUDA avec compilation séparée
│   ├── CMakeLists.txt
│   ├── kernel.cu
│   ├── kernel.h
│   └── src.cpp
└── .vscode/               # Configuration VS Code
    ├── settings.json      # Paramètres CMake et de construction
    └── launch.json        # Configurations de débogage
```

## Extensions VS Code pour le Développement CUDA

### Extensions Requises

1. **C/C++ Extension Pack** (`ms-vscode.cpptools-extension-pack`)
   - Fournit IntelliSense, débogage et navigation de code
   - Inclut les thèmes C/C++ et CMake Tools

2. **CMake Tools** (`ms-vscode.cmake-tools`)
   - Intégration CMake avec VS Code
   - Construire, configurer et déboguer les projets CMake

3. **Nsight Visual Studio Code Edition** (`nvidia.nsight-vscode-edition`)
   - Support de débogage CUDA
   - Débogage des noyaux GPU
   - Intégration CUDA-GDB

4. **Catch2 Test Adapter** (`matepek.vscode-catch2-test-adapter`)
   - Exécuter et déboguer les tests Catch2 depuis VS Code
   - Intégration de l'explorateur de tests
   - Indicateurs visuels d'état des tests

### Extensions Recommandées

5. **CUDA C++** (`kriegalex.vscode-cuda`)
   - Coloration syntaxique CUDA
   - Extraits de code pour la programmation CUDA

6. **C/C++ Snippets** (`hars.cppsnippets`)
   - Extraits de code utiles pour le développement C/C++

7. **Better C++ Syntax** (`jeff-hykin.better-cpp-syntax`)
   - Coloration syntaxique améliorée pour le C++ moderne

8. **Clangd** (`llvm-vs-code-extensions.vscode-clangd`)
   - Alternative à Microsoft C++ IntelliSense
   - Meilleur support pour les fonctionnalités spécifiques à Clang
   - Complétion de code et diagnostics plus précis

9. **LLDB DAP** (`llvm-vs-code-extensions.lldb-dap`)
   - Intégration du débogueur LLDB
   - Meilleure expérience de débogage avec le code compilé par Clang

### Extensions Optionnelles

10. **Error Lens** (`usernamehw.errorlens`)
    - Affiche les erreurs et avertissements en ligne

11. **GitLens** (`eamodio.gitlens`)
    - Intégration Git améliorée

12. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Vérification orthographique pour le code et les commentaires

## Installation des Extensions VS Code

### Méthode 1 : Scripts d'Installation Automatisés

**Linux/macOS :**
```bash
./install-vscode-extensions.sh
```

**Windows :**
```batch
install-vscode-extensions.bat
```

### Méthode 2 : Via l'Interface VS Code
1. Ouvrir VS Code
2. Cliquer sur l'icône Extensions (Ctrl+Shift+X)
3. Rechercher chaque extension par nom
4. Cliquer sur Installer

### Méthode 3 : Ligne de Commande
```bash
# Installer les extensions requises
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition
code --install-extension ms-vscode.cmake-tools
code --install-extension matepek.vscode-catch2-test-adapter

# Installer les extensions recommandées
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension llvm-vs-code-extensions.lldb-dap

# Installer les extensions optionnelles
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker
```

## Débogage

### Débogage CUDA dans VS Code

1. Définir des points d'arrêt dans votre code CUDA (fichiers .cu)
2. Appuyez sur `F5` ou allez dans Exécuter et Déboguer
3. Sélectionnez la configuration "CUDA C++: Launch (cuda-gdb)"
4. Le débogueur s'arrêtera aux points d'arrêt dans le code hôte et périphérique

### Configurations de Débogage Disponibles

- **CUDA C++: Launch (cuda-gdb)** : Déboguer le code CUDA avec cuda-gdb
- Utilise la sélection de cible actuelle de CMake
- Supporte les plateformes Linux et Windows

## Fichiers de Configuration

### `.vscode/settings.json`
Configure CMake pour utiliser Ninja et Clang :
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
Définit les préréglages de construction pour différentes configurations :
- `default` : Construction Debug avec Clang et Ninja
- `release` : Construction Release avec optimisations

### `.vscode/launch.json`
Configurations de débogage pour les applications CUDA avec chemins de débogueur spécifiques à la plateforme.

## Dépannage

### Configuration Spécifique à l'Environnement

Lors du changement de votre environnement de développement ou version CUDA, mettez à jour les chemins suivants :

#### 1. Mettre à jour debuggerPath dans `.vscode/launch.json`
Le chemin du débogueur doit correspondre à votre version et emplacement d'installation CUDA :
```json
"windows": {
    "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/cuda-gdb.exe"
    // Mettre à jour vers v13.0 ou votre version installée :
    // "debuggerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuda-gdb.exe"
}
```

Pour Linux/WSL :
```json
"linux": {
    "debuggerPath": "/usr/bin/cuda-gdb"
    // Ou si installé dans un emplacement différent :
    // "debuggerPath": "/usr/local/cuda-13.0/bin/cuda-gdb"
}
```

#### 2. Vérifier le Chemin CUDA Toolkit
Assurez-vous que votre PATH système inclut la version CUDA correcte :
- Windows : `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`
- Linux : `/usr/local/cuda-13.0/bin`

### Avertissement Architecture CUDA
Si vous voyez "Cannot find valid GPU for '-arch=native'", cela signifie que CMake ne peut pas détecter votre architecture GPU. Vous pouvez la spécifier manuellement dans `CMakeLists.txt` :
```cmake
set(CMAKE_CUDA_ARCHITECTURES "75")  # Pour GTX 1660 Ti, RTX 2060-2080
set(CMAKE_CUDA_ARCHITECTURES "86")  # Pour RTX 3060-3090
set(CMAKE_CUDA_ARCHITECTURES "89")  # Pour RTX 4090
# Ou pour construire sans GPU (supporte plusieurs architectures) :
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
```

### Compatibilité Version Clang
CUDA peut ne pas supporter officiellement les dernières versions de Clang. Le projet utilise le drapeau `-allow-unsupported-compiler` pour contourner les vérifications de version. Si vous rencontrez des problèmes, envisagez d'utiliser GCC à la place :
```bash
cmake -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
```

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.