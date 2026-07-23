@echo off
rem ==========================================================================
rem  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
rem  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
rem  SPDX-License-Identifier: Apache-2.0
rem ==========================================================================
rem Build Release tacuda.dll with MSVC + Ninja + the CUDA toolkit.
rem
rem Prerequisites:
rem   - Visual Studio 2022 (any edition) with C++ workload
rem   - CUDA toolkit 12.x; set CUDA_PATH to its root if the installer did
rem     not (e.g. set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9)
rem
rem Run from the repository root. Output: build-win\tacuda.dll. Copy it to:
rem   bindings\csharp\runtimes\win-x64\native\tacuda.dll  (before dotnet pack)
rem   build\Release\tacuda.dll                            (local Python/Node tests)
setlocal
set "REPO=%~dp0.."

if "%CUDA_PATH%"=="" (
  echo CUDA_PATH is not set. Point it to your CUDA toolkit root, e.g.:
  echo   set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
  exit /b 1
)

rem Locate vcvars64.bat via vswhere (works for Community/Professional/Enterprise).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe not found - is Visual Studio 2022 installed?
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -S "%REPO%" -B "%REPO%\build-win" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DBUILD_TESTING=OFF || exit /b 1
rem All targets (library + optimizer + example_app): keeps MSVC-only
rem breakage outside the DLL visible locally, same as CI.
cmake --build "%REPO%\build-win" -j 12
