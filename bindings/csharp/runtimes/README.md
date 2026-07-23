# Native runtimes

The native libraries are intentionally NOT tracked in git. Before running
`dotnet pack` (or consuming the project locally), place them here:

```
runtimes/win-x64/native/tacuda.dll
runtimes/linux-x64/native/libtacuda.so
```

How to obtain them:

- **CI**: the `native-build` workflow builds both and `release-packages`
  downloads them into this folder automatically before packing.
- **Windows, locally**: `scripts\build_win_dll.bat` (requires MSVC + CUDA
  toolkit, see the script header), then copy `build-win\tacuda.dll` here.
- **Linux, locally**: standard CMake build (`cmake -S . -B build && cmake
  --build build`), then copy `build/libtacuda.so` here.

Verify the export surface after copying:

```
python scripts/check_dll_exports.py runtimes/win-x64/native/tacuda.dll runtimes/linux-x64/native/libtacuda.so
```
