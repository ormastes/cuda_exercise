#!/bin/bash

# VS Code Extensions Installer for CUDA Development
# Run this script to install all recommended VS Code extensions

echo "Installing VS Code extensions for CUDA development..."
echo "=================================================="

# Required extensions
echo ""
echo "Installing required extensions..."
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension nvidia.nsight-vscode-edition

# Recommended extensions
echo ""
echo "Installing recommended extensions..."
code --install-extension twxs.cmake
code --install-extension kriegalex.vscode-cuda
code --install-extension hars.cppsnippets
code --install-extension jeff-hykin.better-cpp-syntax
code --install-extension llvm-vs-code-extensions.vscode-clangd

# Optional extensions
echo ""
echo "Installing optional extensions..."
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension streetsidesoftware.code-spell-checker

echo ""
echo "=================================================="
echo "All extensions have been installed successfully!"
echo "Please restart VS Code to ensure all extensions are loaded."