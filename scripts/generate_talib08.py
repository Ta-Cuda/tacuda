#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Import pinned TA-Lib batch formulas and emit the private CUDA port.

An ordinary regeneration uses the curated, licensed batch-only snapshot.
--import-upstream is a deliberate refresh and requires the exact pinned Git
commit. No build or code generation downloads a floating upstream branch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/ta_lib_0_8_3'
PIN = '62021c1d14ef5499505af5547384cab509adf3fc'
EXTRA = 'MA MAVP MACDEXT APO PPO BBANDS STOCH STOCHF STOCHRSI'.split()

def mask_comments(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                  lambda m: re.sub(r'[^\n]', ' ', m[0]), text, flags=re.S)

def definitions(text):
    masked = mask_comments(text)
    pattern = re.compile(r'^[ \t]*(?:(?:TA_NOINLINE|TA_LIB_API|static|TA_FMA_MULTIVERSION|TA_FMA_STEP_INLINE|TA_FMA_OPEN_CLONE|TA_FMA_OPEN_PLAIN)\s+)*(?:unsigned\s+int|int|double|void|TA_RetCode|TA_MAType|TA_Real)\s+(TA_\w+)\s*\([^;{}]*\)\s*\{', re.M)
    for match in pattern.finditer(masked):
        end, depth = match.end(), 1
        while depth:
            depth += (masked[end] == '{') - (masked[end] == '}')
            end += 1
        yield match[1], text[match.start():end].strip()

def import_upstream(upstream):
    sha = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != PIN:
        raise SystemExit(f'expected pinned TA-Lib {PIN}, got {sha}')
    VENDOR.mkdir(parents=True, exist_ok=True)
    funcs, files = {}, {}
    for path in (upstream / 'src/ta_func').glob('*.c'):
        text = path.read_text()
        for name, code in definitions(text):
            funcs[name] = code
            files[name] = path
    xml = ET.parse(upstream / 'ta_func_api.xml')
    old = subprocess.check_output(['git', '-C', str(upstream), 'show', 'v0.7.1:ta_func_list.txt'], text=True)
    legacy = {line.split()[0] for line in old.splitlines() if line.strip()}
    latest = {f.findtext('Abbreviation') for f in xml.findall('FinancialFunction')}
    seeds = sorted((latest - legacy) | set(EXTRA))
    needed, todo = set(), ['TA_' + n + suffix for n in seeds for suffix in ['', '_Lookback']]
    while todo:
        name = todo.pop()
        if name in needed:
            continue
        needed.add(name)
        code = mask_comments(funcs[name])
        todo.extend(set(re.findall(r'\bTA_\w+(?=\s*\()', code)) & set(funcs) - needed)
    by_file = {}
    for name in sorted(needed):
        by_file.setdefault(files[name], []).append(funcs[name])
    hashes = {}
    for source, codes in sorted(by_file.items()):
        text = source.read_text()
        license_header = text[:text.index('*/') + 2]
        payload = license_header + '\n\n/* Pinned batch-only extraction; see PROVENANCE.json. */\n\n' + '\n\n'.join(codes) + '\n'
        (VENDOR / source.name).write_text(payload, newline='\n')
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (VENDOR / 'ta_defs.h').write_bytes((upstream / 'include/ta_defs.h').read_bytes())
    utility = (upstream / 'src/ta_func/ta_utility.h').read_text()
    begin = utility.index('#define HILBERT_VARIABLES')
    end = utility.index('#define PER_TO_K', begin)
    (VENDOR / 'hilbert_macros.h').write_text(utility[:utility.index('*/')+2] + '\n' + utility[begin:end], newline='\n')
    (VENDOR / 'LICENSE').write_bytes((upstream / 'LICENSE').read_bytes())
    catalog = []
    for f in xml.findall('FinancialFunction'):
        name = f.findtext('Abbreviation')
        signature = funcs['TA_' + name] if 'TA_' + name in needed else ''
        catalog.append({'name': name, 'legacy': name in legacy,
            'inputs': [a.findtext('Name') for a in f.findall('RequiredInputArguments/RequiredInputArgument')],
            'outputs': [{'name': a.findtext('Name'), 'ctype': 'int' if a.findtext('Type') == 'Integer Array' else 'float'} for a in f.findall('OutputArguments/OutputArgument')],
            'options': [{'name': a.findtext('Name'), 'type': a.findtext('Type'), 'default': a.findtext('DefaultValue'),
                         'minimum': a.findtext('Range/Minimum'), 'maximum': a.findtext('Range/Maximum')} for a in f.findall('OptionalInputArguments/OptionalInputArgument')],
            'signature': signature[:signature.find('{')].strip() if signature else ''})
    (VENDOR / 'catalog.json').write_text(json.dumps(catalog, indent=2) + '\n', newline='\n')
    (VENDOR / 'PROVENANCE.json').write_text(json.dumps({'version':'0.8.3','commit':PIN,'function_count':237,'source_sha256':hashes}, indent=2)+'\n', newline='\n')

def render():
    functions = dict(item for path in sorted(VENDOR.glob('ta_*.c')) for item in definitions(path.read_text()))
    names = sorted(functions, key=len, reverse=True)
    call = re.compile(r'\b(' + '|'.join(map(re.escape, names)) + r')\s*\(')
    adapted = []
    for name in sorted(functions):
        code = re.sub(r'/\*.*?\*/|//[^\n]*', '', functions[name], flags=re.S)
        code = re.sub(r'\b(?:TA_LIB_API|TA_NOINLINE|TA_FMA_MULTIVERSION|static)\s*', '', code)
        code = call.sub(lambda m: m[1] + '(arena, ', code)
        # The first occurrence is the definition; remaining ones are calls.
        code = code.replace(name + '(arena, ', name + '(Arena& arena, ', 1)
        code = re.sub(r'\(Arena& arena,\s*void\s*\)', '(Arena& arena)', code)
        code = re.sub(r'\(arena,\s*\)', '(arena)', code)
        code = re.sub(r'\bTA_MAType\b', 'int', code)
        # Select the CUDA double math overload rather than a host-only C++
        # integral overload. This does not change the mathematical argument.
        code = re.sub(r'\b(atan|sqrt|log|exp|sin|cos)\((\d+)\)', r'\1(\2.0)', code)
        code = re.sub(r'\b(?:TA_Malloc|malloc)\s*\(', 'arena.allocate(', code)
        code = re.sub(r'\b(?:TA_Free|free)\s*\(', 'arena.release(', code)
        code = '\n'.join(line.rstrip() for line in code.splitlines() if line.strip())
        adapted.append('static TA08_HD ' + code)
    prototypes = [x[:x.index('{')].rstrip() + ';' for x in adapted]
    first = next(iter(sorted(VENDOR.glob('ta_*.c')))).read_text()
    license_header = first[:first.index('*/')+2]
    return (license_header + '\n\n// Generated by scripts/generate_talib08.py from licensed TA-Lib 0.8.3.\n'
            '// Host/device formulas; only allocation and context plumbing are adapted.\n'
            + '\n'.join(prototypes) + '\n\n' + '\n\n'.join(adapted) + '\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--import-upstream', type=Path)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.import_upstream:
        import_upstream(args.import_upstream)
    from talib08_catalog import ENTRIES
    catalog = json.loads((VENDOR / 'catalog.json').read_text())
    enum = '// Generated by scripts/generate_talib08.py.\nenum class Function {\n'
    enum += ''.join('    '+e['name']+',\n' for e in ENTRIES) + '    Count\n};\n'
    dispatch = ['// Generated by scripts/generate_talib08.py.',
        'struct Info { int inputs, outputs, integerMask; };',
        'static constexpr Info infos[] = {']
    for e in ENTRIES:
        mask = sum((1 << i) for i,t in enumerate(e['output_dtypes']) if t == 'int')
        dispatch.append(f"    {{{len(e['inputs'])}, {len(e['outputs'])}, {mask}}}, // {e['name']}")
    dispatch += ['};', 'TA08_HD int lookback(Function f, ref::Arena& arena, const Params& params) {', 'switch(f) {']
    for e in ENTRIES:
        options = ', '.join(f"static_cast<{t}>(params.value[{i}])" for i,(t,n) in enumerate(e['scalars']))
        dispatch.append(f"case Function::{e['name']}: return ref::TA_{e['name']}_Lookback(arena{', ' if options else ''}{options});")
    dispatch += ['default: return -1;', '}}',
        'TA08_HD TA_RetCode evaluate(Function f, ref::Arena& arena, const double* const* in, void* const* out, int length, int* begin, int* count, const Params& params) {', 'switch(f) {']
    for e in ENTRIES:
        values = ['arena','0','length - 1'] + [f'in[{i}]' for i in range(len(e['inputs']))]
        values += [f"static_cast<{t}>(params.value[{i}])" for i,(t,n) in enumerate(e['scalars'])]
        values += ['begin','count'] + [f"static_cast<{'int' if t=='int' else 'double'}*>(out[{i}])" for i,t in enumerate(e['output_dtypes'])]
        dispatch.append(f"case Function::{e['name']}: return ref::TA_{e['name']}({', '.join(values)});")
    dispatch += ['default: return TA_BAD_PARAM;', '}}']
    payloads = {ROOT / 'src/compat/talib08_formulas.inc':render(),
                ROOT / 'include/utils/Talib08Functions.inc':enum,
                ROOT / 'src/compat/talib08_dispatch.inc':'\n'.join(dispatch)+'\n'}
    payloads[ROOT / 'tests/reference/talib-0.8.3.json'] = json.dumps({
        'native_version':'0.8.3','python_version':'0.8.3','commit':PIN,
        'unstable_period':0,'functions':sorted(e['name'] for e in catalog)},indent=2)+'\n'
    notices = ('TaCuda third-party notices\n\n'
        'The native library includes a CUDA adaptation of TA-Lib 0.8.3 batch formulas.\n'
        f'Upstream: https://github.com/TA-Lib/ta-lib/tree/{PIN}\n'
        'The following notice applies to that component; the TaCuda project license is unchanged.\n\n'
        + (VENDOR / 'LICENSE').read_text())
    for path in ['THIRD_PARTY_NOTICES.txt','bindings/python/tacuda/THIRD_PARTY_NOTICES.txt',
                 'bindings/nodejs/THIRD_PARTY_NOTICES.txt']:
        payloads[ROOT / path] = notices
    for path, expected in payloads.items():
        if args.check:
            if not path.exists() or path.read_text() != expected:
                raise SystemExit(f'{path} needs regeneration')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, newline='\n')
    print('TA-Lib 0.8.3 formulas: current')

if __name__ == '__main__':
    main()
