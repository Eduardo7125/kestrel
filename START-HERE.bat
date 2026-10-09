@echo off
rem Kestrel: from a fresh checkout to a running model in one step.
rem Double-click this file, or run it from a terminal with options for
rem `kestrel setup` (for example: START-HERE.bat --list).
setlocal
cd /d "%~dp0"

if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where cargo >nul 2>nul
if not errorlevel 1 goto build

echo Kestrel is written in Rust and is built on this machine once.
echo The Rust toolchain is not installed.
where winget >nul 2>nul
if errorlevel 1 goto norust
choice /M "Install Rust now with winget"
if errorlevel 2 goto norust
winget install -e --id Rustlang.Rustup
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where cargo >nul 2>nul
if errorlevel 1 goto norust

:build
if not exist target\release\kestrel.exe echo Building Kestrel (the first build takes a few minutes)...
cargo build --release --quiet -p kestrel-cli
if errorlevel 1 goto buildfail
target\release\kestrel.exe setup %*
goto end

:norust
echo Install Rust from https://rustup.rs, then run START-HERE.bat again.
goto end

:buildfail
echo.
echo The build failed. If the error mentions link.exe, install the
echo "Desktop development with C++" workload of the Visual Studio Build Tools:
echo   winget install Microsoft.VisualStudio.2022.BuildTools
echo then run START-HERE.bat again.

:end
pause
endlocal
