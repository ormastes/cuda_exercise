@echo off
REM VS Code Extensions Installer for CUDA Development
REM Run this script to install all recommended VS Code extensions

echo Installing VS Code extensions for CUDA development...
echo ==================================================

REM Required extensions
echo.
echo Installing required extensions...
call code --install-extension ms-vscode.cpptools-extension-pack
call code --install-extension nvidia.nsight-vscode-edition
call code --install-extension ms-vscode.cmake-tools
call code --install-extension matepek.vscode-catch2-test-adapter

REM Recommended extensions
echo.
echo Installing recommended extensions...
call code --install-extension kriegalex.vscode-cuda
call code --install-extension hars.cppsnippets
call code --install-extension jeff-hykin.better-cpp-syntax
call code --install-extension llvm-vs-code-extensions.vscode-clangd
call code --install-extension llvm-vs-code-extensions.lldb-dap

REM Optional extensions
echo.
echo Installing optional extensions...
call code --install-extension usernamehw.errorlens
call code --install-extension eamodio.gitlens
call code --install-extension streetsidesoftware.code-spell-checker

echo.
echo ==================================================
echo All extensions have been installed successfully!
echo Please restart VS Code to ensure all extensions are loaded.
pause