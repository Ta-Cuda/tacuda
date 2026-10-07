#!/usr/bin/env python3
"""Check every public C declaration against PE/ELF exports, without loading CUDA."""
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADERS = ('tacuda.h', 'tacuda_device.h', 'tacuda_stream.h')


def required_exports():
    names = set()
    for name in HEADERS:
        text = (ROOT / 'include' / name).read_text(encoding='utf-8')
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
        names.update(re.findall(r'CTAPI_EXPORT\s+[^;{}]*?\b(ct_\w+)\s*\(', text))
    if not names:
        raise ValueError('no public declarations found')
    return names


def dumpbin_path():
    tool = shutil.which('dumpbin')
    if tool:
        return tool
    vswhere = pathlib.Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    install = subprocess.check_output(
        [str(vswhere), '-latest', '-products', '*', '-requires',
         'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
        text=True).strip()
    candidates = sorted((pathlib.Path(install) / 'VC/Tools/MSVC').glob('*/bin/Hostx64/x64/dumpbin.exe'))
    if not candidates:
        raise FileNotFoundError('dumpbin not found; initialize the MSVC environment')
    return str(candidates[-1])


def exported_symbols(path):
    with path.open('rb') as fh:
        magic = fh.read(4)
    if magic[:2] == b'MZ':
        output = subprocess.check_output([dumpbin_path(), '/nologo', '/exports', str(path)], text=True, errors='replace')
        return set(re.findall(r'^\s+\d+\s+[\da-fA-F]+\s+[\da-fA-F]+\s+(ct_\w+)(?=\s|$)', output, re.M))
    if magic == b'\x7fELF':
        output = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True, errors='replace')
        return set(re.findall(r'^\S+\s+\S\s+(ct_\w+)(?:@\S+)?$', output, re.M))
    raise ValueError('expected a PE or ELF native library')


def check(path, required=None):
    try:
        required = required if required is not None else required_exports()
        exported = exported_symbols(path)
        missing = sorted(required - exported)
        if missing:
            print(f'FAIL {path}: {len(missing)} missing exports: ' + ', '.join(missing))
            return False
        print(f'PASS {path}: all {len(required)} public declarations exported')
        return True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'FAIL {path}: {exc}')
        return False


def main():
    targets = [pathlib.Path(p) for p in sys.argv[1:]]
    if not targets:
        targets = [ROOT / ('build-win/tacuda.dll' if os.name == 'nt' else 'build/libtacuda.so')]
    return 0 if all([check(path) for path in targets]) else 1


if __name__ == '__main__':
    sys.exit(main())
