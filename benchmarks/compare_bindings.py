#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Compare Python and C# bindings by running both test suites.
"""

import subprocess
import sys
import os

def main():
    print("=" * 70)
    print("TACUDA Bindings Comparison: Python vs C#")
    print("=" * 70)
    
    # Set up paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll_path = os.path.join(project_root, "build", "Release")
    python_bindings = os.path.join(project_root, "bindings", "python")
    csharp_project = os.path.join(project_root, "bindings", "csharp", "ConsoleExample")
    
    # Check if DLL exists
    dll_file = os.path.join(dll_path, "tacuda.dll")
    if not os.path.exists(dll_file):
        print(f"\n✗ tacuda.dll not found at {dll_file}")
        print("  Please build the project first: cmake --build build --config Release")
        return 1
    
    print(f"\n✓ Found tacuda.dll at {dll_path}")
    
    # =========================================================================
    # Python Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("PYTHON BINDINGS TEST")
    print("=" * 70)
    
    env = os.environ.copy()
    env["TACUDA_LIBRARY_PATH"] = dll_path
    env["PYTHONPATH"] = python_bindings
    
    python_script = os.path.join(os.path.dirname(__file__), "test_bindings_comparison.py")
    
    try:
        result = subprocess.run(
            [sys.executable, python_script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        python_success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print("✗ Python tests timed out")
        python_success = False
    except Exception as e:
        print(f"✗ Python tests failed: {e}")
        python_success = False
    
    # =========================================================================
    # C# Tests
    # =========================================================================
    print("\n" + "=" * 70)
    print("C# BINDINGS TEST")
    print("=" * 70)
    
    # Check if .NET is available
    try:
        dotnet_check = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
        if dotnet_check.returncode != 0:
            print("✗ .NET SDK not found")
            csharp_success = False
        else:
            print(f"✓ .NET SDK version: {dotnet_check.stdout.strip()}")
            
            # Copy DLL to output directory
            csharp_output = os.path.join(csharp_project, "bin", "Release", "net7.0")
            os.makedirs(csharp_output, exist_ok=True)
            
            import shutil
            dest_dll = os.path.join(csharp_output, "tacuda.dll")
            if not os.path.exists(dest_dll):
                shutil.copy(dll_file, dest_dll)
                print(f"✓ Copied tacuda.dll to {csharp_output}")
            
            # Build and run C# tests
            try:
                build_result = subprocess.run(
                    ["dotnet", "build", "-c", "Release"],
                    cwd=csharp_project,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if build_result.returncode != 0:
                    print("✗ C# build failed:")
                    print(build_result.stderr)
                    csharp_success = False
                else:
                    print("✓ C# project built successfully")
                    
                    # Run tests
                    run_result = subprocess.run(
                        ["dotnet", "run", "-c", "Release", "--", "--test"],
                        cwd=csharp_project,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env={**os.environ, "PATH": dll_path + os.pathsep + os.environ.get("PATH", "")}
                    )
                    
                    print(run_result.stdout)
                    if run_result.stderr:
                        print("STDERR:", run_result.stderr)
                    csharp_success = run_result.returncode == 0
                    
            except subprocess.TimeoutExpired:
                print("✗ C# tests timed out")
                csharp_success = False
            except Exception as e:
                print(f"✗ C# tests failed: {e}")
                csharp_success = False
                
    except FileNotFoundError:
        print("✗ .NET SDK not installed")
        csharp_success = False
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\nPython Bindings: {'✓ PASSED' if python_success else '✗ FAILED'}")
    print(f"C# Bindings:     {'✓ PASSED' if csharp_success else '✗ FAILED'}")
    
    if python_success and csharp_success:
        print("\n✓ Both bindings are working correctly!")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

