// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
// Real native round-trip against the pinned CPU fixture; no mock FFI.
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const t = require(process.env.TACUDA_NODE_BINDINGS || '../../bindings/nodejs');
const f = JSON.parse(fs.readFileSync(path.join(__dirname, '../fixtures/talib08_typed.json')));
const check = (got, expected) => {
  assert.equal(got.length, expected.length);
  for (let i=0;i<got.length;i++) {
    if (expected[i] === null) assert(Number.isNaN(got[i]), `expected NaN at ${i}`);
    else assert(Math.abs(got[i]-expected[i]) <= 2e-5*Math.max(1,Math.abs(expected[i])), `${i}: ${got[i]} vs ${expected[i]}`);
  }
};
const h=t.ha(f.open,f.high,f.low,f.close);
for (const [name,expected] of Object.entries(f.ha)) check(h[name],expected);
const fractal=t.fractal(f.high,f.low,2,2);
assert(fractal.swingHigh instanceof Int32Array);
assert(fractal.swingLow instanceof Int32Array);
assert.deepEqual(Array.from(fractal.swingHigh),f.fractal.swingHigh);
assert.deepEqual(Array.from(fractal.swingLow),f.fractal.swingLow);
const trend=t.supertrend(f.high,f.low,f.close,5,1.5);
assert(trend.trend instanceof Int32Array);
check(trend.supertrend,f.supertrend.real);
assert.deepEqual(Array.from(trend.trend),f.supertrend.trend);
const batch=t.supertrendBatch([f.high,f.high],[f.low,f.low],[f.close,f.close],5,1.5);
for (let row=0;row<2;row++) {
  assert(batch.trend[row] instanceof Int32Array);
  assert.deepEqual(Array.from(batch.trend[row]),f.supertrend.trend);
  check(batch.supertrend[row],f.supertrend.real);
}
check(t.rma(f.close,14),f.rma);
console.log('PASS Node: four outputs, int32 flags, mixed single/batch outputs, double options and RMA');
