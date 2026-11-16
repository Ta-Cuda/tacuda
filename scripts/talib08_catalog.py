# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Pinned API descriptors shared by emitters and the versioned oracle tests."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'third_party/ta_lib_0_8_3/catalog.json').read_text())
RENAMES = {'ASI':'asi_wilder', 'KURTOSIS':'kurtosis_unbiased', 'PVO':'pvo_ma'}
RENAMES.update({'BBANDS':'bbands_ma','STOCH':'stochastic_ma','STOCHF':'stochf_ma','STOCHRSI':'stochrsi_ma'})
RENAMES.update({'APO':'apo_ma','PPO':'ppo_ma'})
EXTRA = {'MA','MAVP','MACDEXT','APO','PPO','BBANDS','STOCH','STOCHF','STOCHRSI'}

def scalar_name(name):
    name = name.removeprefix('optIn')
    if name == 'TimePeriod': return 'period'
    if name == 'MAType': return 'maType'
    name = re.sub(r'^[A-Z]+(?=[A-Z][a-z]|[0-9]|$)', lambda m:m[0].lower(), name)
    name = name[0].lower() + name[1:]
    return name.replace('_', '')

def input_name(name):
    return {'inReal':'inputs','inHigh':'highs','inLow':'lows','inOpen':'opens',
            'inClose':'closes','inVolume':'volumes','inPeriods':'periods'}[name]

def output_name(name):
    if name == 'outReal': return 'outputs'
    name = name.removeprefix('out')
    name = re.sub(r'^[A-Z]+(?=[A-Z][a-z]|$)', lambda m:m[0].lower(), name)
    return name[0].lower() + name[1:]

def entries():
    result = []
    for record in CATALOG:
        if record['legacy'] and record['name'] not in EXTRA: continue
        signature = record['signature']
        arguments = re.search(r'\((.*)\)', signature, re.S)[1].split(',')
        ins, outs, scalars = [], [], []
        for argument in arguments[2:]:
            argument = argument.strip()
            name = re.search(r'(\w+)\s*(?:\[\])?$', argument)[1]
            if name in ['outBegIdx','outNBElement']: continue
            if argument.startswith('const '): ins.append(input_name(name))
            elif '[]' in argument:
                outs.append((output_name(name), 'int' if argument.startswith('int ') else 'float'))
            else: scalars.append(('double' if argument.startswith('double ') else 'int', scalar_name(name)))
        result.append({**record, 'api':RENAMES.get(record['name'],record['name'].lower()),
            'inputs':tuple(ins), 'outputs':tuple(n for n,t in outs),
            'output_dtypes':tuple(t for n,t in outs), 'scalars':tuple(scalars),
            'public_new':(not record['legacy'] and record['name'] not in ['NVI','PVI']) or record['name'] in ['APO','PPO','BBANDS','STOCH','STOCHF','STOCHRSI']})
    return result

ENTRIES = entries()
NEW = [e for e in ENTRIES if e['public_new']]
