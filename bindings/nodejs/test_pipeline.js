#!/usr/bin/env node
// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
/**
 * Comprehensive test: exercises every exported indicator in the Node.js bindings.
 * Similar to C++ test_context_pipeline.cpp, C# TestBindings.cs, Python test_pipeline.py
 */

const tacuda = require('./index');

const EPS = 1e-6;

// ============================================================================
// Helpers
// ============================================================================

function generateOHLCV(size) {
    const open = new Float32Array(size);
    const high = new Float32Array(size);
    const low  = new Float32Array(size);
    const close = new Float32Array(size);
    const volume = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
        const t = i;
        const base = 100.0 + 2.0 * Math.sin(0.01 * t) + 0.5 * Math.cos(0.005 * t) + 0.0001 * t;
        const o = base + 0.1 * Math.sin(0.03 * t);
        const c = base + 0.1 * Math.cos(0.02 * t);
        const h = Math.max(o, c) + 0.2 + 0.3 * Math.abs(Math.sin(0.01 * t));
        const l = Math.min(o, c) - 0.2 - 0.3 * Math.abs(Math.cos(0.01 * t));
        open[i] = o; high[i] = h; low[i] = l; close[i] = c;
        volume[i] = 100000.0 + ((i * 37) % 1000) * 10.0;
    }
    return { open, high, low, close, volume };
}

function validateRange(arr, lo, hi, name) {
    for (let i = 0; i < arr.length; i++) {
        if (!Number.isNaN(arr[i]) && (arr[i] < lo - EPS || arr[i] > hi + EPS)) {
            console.log(`    ${name} out of range [${lo},${hi}]: ${arr[i]} at i=${i}`);
            return false;
        }
    }
    return true;
}

function validateNonNeg(arr, name) {
    for (let i = 0; i < arr.length; i++) {
        if (!Number.isNaN(arr[i]) && arr[i] < -EPS) {
            console.log(`    ${name} negative: ${arr[i]} at i=${i}`);
            return false;
        }
    }
    return true;
}

function hasValidValues(arr) {
    for (let i = 0; i < arr.length; i++) {
        if (!Number.isNaN(arr[i])) return true;
    }
    return false;
}

const results = { passed: 0, failed: 0 };

function test(name, fn) {
    try {
        const ok = fn();
        if (ok) {
            console.log(`  [PASS] ${name}`);
            results.passed++;
        } else {
            console.log(`  [FAIL] ${name}`);
            results.failed++;
        }
    } catch (e) {
        console.log(`  [ERROR] ${name}: ${e.message}`);
        results.failed++;
    }
}

// ============================================================================
// Test Suite
// ============================================================================

function runAllTests() {
    console.log('='.repeat(70));
    console.log('TACUDA Node.js Comprehensive Indicator Test');
    console.log('='.repeat(70));

    const N = 4096;
    const { open, high, low, close, volume } = generateOHLCV(N);

    console.log(`Data size: ${N} samples\n`);

    // ---- Moving Averages ----
    console.log('--- Moving Averages ---');
    test('SMA(20)',   () => hasValidValues(tacuda.sma(close, 20)));
    test('EMA(12)',   () => hasValidValues(tacuda.ema(close, 12)));
    test('WMA(10)',   () => hasValidValues(tacuda.wma(close, 10)));
    test('DEMA(10)',  () => hasValidValues(tacuda.dema(close, 10)));
    test('TEMA(10)',  () => hasValidValues(tacuda.tema(close, 10)));
    test('TRIMA(10)', () => hasValidValues(tacuda.trima(close, 10)));
    test('T3(5,0.7)', () => hasValidValues(tacuda.t3(close, 5, 0.7)));
    test('KAMA(10,2,30)', () => hasValidValues(tacuda.kama(close, 10, 2, 30)));
    test('MA(20, SMA)', () => hasValidValues(tacuda.ma(close, 20, 0)));
    test('MAVP', () => {
        const periods = new Float32Array(N);
        for (let i = 0; i < N; i++) periods[i] = 10 + (i % 10);
        return hasValidValues(tacuda.mavp(close, periods, 5, 30, 0));
    });

    // ---- Momentum ----
    console.log('--- Momentum ---');
    test('Momentum(10)', () => hasValidValues(tacuda.momentum(close, 10)));
    test('Change(10)',   () => hasValidValues(tacuda.change(close, 10)));
    test('ROC(10)',      () => hasValidValues(tacuda.roc(close, 10)));
    test('ROCP(10)',     () => hasValidValues(tacuda.rocp(close, 10)));
    test('ROCR(10)',     () => hasValidValues(tacuda.rocr(close, 10)));
    test('ROCR100(10)',  () => hasValidValues(tacuda.rocr100(close, 10)));
    test('TRIX(14)',     () => hasValidValues(tacuda.trix(close, 14)));
    test('CMO(14)',      () => hasValidValues(tacuda.cmo(close, 14)));
    test('RSI(14)',      () => validateRange(tacuda.rsi(close, 14), 0, 100, 'RSI'));
    test('APO(12,26)',   () => hasValidValues(tacuda.apo(close, 12, 26)));
    test('PPO(12,26)',   () => hasValidValues(tacuda.ppo(close, 12, 26)));
    test('PVO(12,26)',   () => hasValidValues(tacuda.pvo(volume, 12, 26)));

    // ---- MACD ----
    console.log('--- MACD ---');
    test('MACD_LINE(12,26)', () => hasValidValues(tacuda.macdLine(close, 12, 26)));
    test('MACD(12,26,9)', () => {
        const r = tacuda.macd(close, 12, 26, 9, 0);
        return hasValidValues(r.macd) && hasValidValues(r.signal) && hasValidValues(r.hist);
    });
    test('MACDFIX(9)', () => {
        const r = tacuda.macdfix(close, 9);
        return hasValidValues(r.macd) && hasValidValues(r.signal) && hasValidValues(r.hist);
    });
    test('MAMA(0.5,0.05)', () => {
        const r = tacuda.mama(close, 0.5, 0.05);
        return hasValidValues(r.mama) && hasValidValues(r.fama);
    });

    // ---- Statistics ----
    console.log('--- Statistics ---');
    test('StdDev(20)',  () => validateNonNeg(tacuda.stddev(close, 20), 'StdDev'));
    test('AvgDev(20)',  () => validateNonNeg(tacuda.avgdev(close, 20), 'AvgDev'));
    test('Variance(20)',() => validateNonNeg(tacuda.variance(close, 20), 'Var'));
    test('Sum(10)',     () => hasValidValues(tacuda.sum(close, 10)));
    test('Max(10)',     () => hasValidValues(tacuda.max(close, 10)));
    test('Min(10)',     () => hasValidValues(tacuda.min(close, 10)));
    test('MaxIndex(10)',() => hasValidValues(tacuda.maxindex(close, 10)));
    test('MinIndex(10)',() => hasValidValues(tacuda.minindex(close, 10)));
    test('Midpoint(10)',() => hasValidValues(tacuda.midpoint(close, 10)));
    test('MinMax(10)', () => {
        const r = tacuda.minmax(close, 10);
        return hasValidValues(r.min) && hasValidValues(r.max);
    });
    test('MinMaxIndex(10)', () => {
        const r = tacuda.minmaxindex(close, 10);
        return hasValidValues(r.minIdx) && hasValidValues(r.maxIdx);
    });
    test('Beta(10)',  () => hasValidValues(tacuda.beta(close, open, 10)));
    test('Correl(10)',() => hasValidValues(tacuda.correl(close, open, 10)));

    // ---- Linear Regression ----
    console.log('--- Linear Regression ---');
    test('LinearReg(14)',         () => hasValidValues(tacuda.linearreg(close, 14)));
    test('LinearReg_Slope(14)',   () => hasValidValues(tacuda.linearregSlope(close, 14)));
    test('LinearReg_Intercept(14)',() => hasValidValues(tacuda.linearregIntercept(close, 14)));
    test('LinearReg_Angle(14)',   () => hasValidValues(tacuda.linearregAngle(close, 14)));
    test('TSF(14)',               () => hasValidValues(tacuda.tsf(close, 14)));

    // ---- Volatility / Bands ----
    console.log('--- Volatility / Bands ---');
    test('BBands(20,2,2)', () => {
        const r = tacuda.bbands(close, 20, 2.0, 2.0);
        return hasValidValues(r.upper) && hasValidValues(r.middle) && hasValidValues(r.lower);
    });
    test('AccBands(20)', () => {
        const r = tacuda.accbands(high, low, close, 20);
        return hasValidValues(r.upper) && hasValidValues(r.middle) && hasValidValues(r.lower);
    });
    test('ATR(14)',    () => validateNonNeg(tacuda.atr(high, low, close, 14, 0.0), 'ATR'));
    test('NATR(14)',   () => validateNonNeg(tacuda.natr(high, low, close, 14), 'NATR'));
    test('TRANGE',     () => validateNonNeg(tacuda.trange(high, low, close), 'TRANGE'));

    // ---- Oscillators ----
    console.log('--- Oscillators ---');
    test('Stochastic(14,3)', () => {
        const r = tacuda.stochastic(high, low, close, 14, 3);
        return validateRange(r.k, 0, 100, 'StochK') && validateRange(r.d, 0, 100, 'StochD');
    });
    test('StochF(5,3)', () => {
        const r = tacuda.stochf(high, low, close, 5, 3);
        return validateRange(r.k, 0, 100, 'FastK') && validateRange(r.d, 0, 100, 'FastD');
    });
    test('StochRSI(14,5,3)', () => {
        const r = tacuda.stochrsi(close, 14, 5, 3);
        return hasValidValues(r.k) && hasValidValues(r.d);
    });
    test('CCI(20)',    () => hasValidValues(tacuda.cci(high, low, close, 20)));
    test('WillR(14)',  () => validateRange(tacuda.willr(high, low, close, 14), -100, 0, 'WillR'));
    test('UltOsc(7,14,28)', () => validateRange(tacuda.ultosc(high, low, close, 7, 14, 28), 0, 100, 'UltOsc'));

    // ---- Directional Movement ----
    console.log('--- Directional Movement ---');
    test('ADX(14)',     () => validateRange(tacuda.adx(high, low, close, 14), 0, 100, 'ADX'));
    test('ADXR(14)',    () => validateRange(tacuda.adxr(high, low, close, 14), 0, 100, 'ADXR'));
    test('DX(14)',      () => validateRange(tacuda.dx(high, low, close, 14), 0, 100.001, 'DX'));
    test('+DM(14)',     () => validateNonNeg(tacuda.plusDm(high, low, 14), '+DM'));
    test('-DM(14)',     () => validateNonNeg(tacuda.minusDm(high, low, 14), '-DM'));
    test('+DI(14)',     () => validateNonNeg(tacuda.plusDi(high, low, close, 14), '+DI'));
    test('-DI(14)',     () => validateNonNeg(tacuda.minusDi(high, low, close, 14), '-DI'));

    // ---- Volume ----
    console.log('--- Volume ---');
    test('OBV',         () => hasValidValues(tacuda.obv(close, volume)));
    test('MFI(14)',     () => validateRange(tacuda.mfi(high, low, close, volume, 14), 0, 100, 'MFI'));
    test('AD',          () => hasValidValues(tacuda.ad(high, low, close, volume)));
    test('ADOSC(3,10)', () => hasValidValues(tacuda.adosc(high, low, close, volume, 3, 10)));
    test('IMI(14)',     () => hasValidValues(tacuda.imi(open, close, 14)));
    test('NVI',         () => hasValidValues(tacuda.nvi(close, volume)));
    test('PVI',         () => hasValidValues(tacuda.pvi(close, volume)));

    // ---- SAR ----
    console.log('--- SAR ---');
    test('SAR(0.02,0.2)',  () => hasValidValues(tacuda.sar(high, low, 0.02, 0.2)));
    test('SAREXT',         () => hasValidValues(tacuda.sarext(high, low)));

    // ---- Aroon ----
    console.log('--- Aroon ---');
    test('Aroon(14)', () => {
        const r = tacuda.aroon(high, low, 14, 14);
        return hasValidValues(r.up) && hasValidValues(r.down) && hasValidValues(r.osc);
    });
    test('AroonOsc(14)', () => hasValidValues(tacuda.aroonosc(high, low, 14)));

    // ---- Price Transform ----
    console.log('--- Price Transform ---');
    test('AvgPrice',   () => hasValidValues(tacuda.avgprice(open, high, low, close)));
    test('MedPrice',   () => hasValidValues(tacuda.medprice(high, low)));
    test('TypPrice',   () => hasValidValues(tacuda.typprice(high, low, close)));
    test('WclPrice',   () => hasValidValues(tacuda.wclprice(high, low, close)));
    test('MidPrice(10)',() => hasValidValues(tacuda.midprice(high, low, 10)));
    test('BOP',        () => hasValidValues(tacuda.bop(open, high, low, close)));

    // ---- Hilbert Transform ----
    console.log('--- Hilbert Transform ---');
    test('HT_DCPERIOD',  () => hasValidValues(tacuda.htDcperiod(close)));
    test('HT_DCPHASE',   () => hasValidValues(tacuda.htDcphase(close)));
    test('HT_TRENDLINE', () => hasValidValues(tacuda.htTrendline(close)));
    test('HT_TRENDMODE', () => hasValidValues(tacuda.htTrendmode(close)));
    test('HT_PHASOR', () => {
        const r = tacuda.htPhasor(close);
        return hasValidValues(r.inphase) && hasValidValues(r.quadrature);
    });
    test('HT_SINE', () => {
        const r = tacuda.htSine(close);
        return hasValidValues(r.sine) && hasValidValues(r.leadsine);
    });

    // ---- Candlestick Patterns ----
    console.log('--- Candlestick Patterns ---');
    for (const name of tacuda.CDL_NAMES) {
        test(`CDL_${name}`, () => {
            const out = tacuda.cdlPattern(name, open, high, low, close);
            return out.length === N;   // just verify it runs without error
        });
    }

    // ---- Vector Math ----
    console.log('--- Vector Math ---');
    const mathInput = new Float32Array(N);
    for (let i = 0; i < N; i++) mathInput[i] = -0.95 + (1.9 * i) / (N - 1);  // [-0.95, 0.95]
    const positiveInput = new Float32Array(N);
    for (let i = 0; i < N; i++) positiveInput[i] = 0.1 + i * 0.01;

    const unaryRef = {
        acos: Math.acos, asin: Math.asin, atan: Math.atan, ceil: Math.ceil,
        cos: Math.cos, cosh: Math.cosh, exp: Math.exp, floor: Math.floor,
        ln: Math.log, log10: Math.log10, sin: Math.sin, sinh: Math.sinh,
        sqrt: Math.sqrt, tan: Math.tan, tanh: Math.tanh,
    };
    const needsPositive = new Set(['ln', 'log10', 'sqrt']);
    for (const name of tacuda.UNARY_MATH_NAMES) {
        test(`${name.toUpperCase()}`, () => {
            const src = needsPositive.has(name) ? positiveInput : mathInput;
            const out = tacuda[name](src);
            for (let i = 0; i < N; i += 97) {
                const expected = Math.fround(unaryRef[name](src[i]));
                if (Math.abs(out[i] - expected) > 1e-4 * Math.max(1, Math.abs(expected))) {
                    console.log(`    ${name}[${i}]: got ${out[i]}, expected ${expected}`);
                    return false;
                }
            }
            return true;
        });
    }
    test('ADD/SUB/MULT/DIV', () => {
        const a = positiveInput, b = mathInput;
        const addOut = tacuda.add(a, b), subOut = tacuda.sub(a, b);
        const multOut = tacuda.mult(a, b), divOut = tacuda.div(a, b);
        for (let i = 0; i < N; i += 61) {
            if (Math.abs(addOut[i] - (a[i] + b[i])) > 1e-4) return false;
            if (Math.abs(subOut[i] - (a[i] - b[i])) > 1e-4) return false;
            if (Math.abs(multOut[i] - a[i] * b[i]) > 1e-4) return false;
            const expDiv = a[i] / b[i];
            if (Number.isFinite(expDiv) && Math.abs(divOut[i] - expDiv) > 1e-3 * Math.max(1, Math.abs(expDiv))) return false;
        }
        return true;
    });

    // ---- MACDEXT ----
    console.log('--- MACDEXT ---');
    test('MACDEXT(12,EMA,26,EMA,9,EMA) == MACD(12,26,9,EMA)', () => {
        const ref = tacuda.macd(close, 12, 26, 9, tacuda.MA_TYPE.EMA);
        const ext = tacuda.macdext(close, 12, tacuda.MA_TYPE.EMA, 26, tacuda.MA_TYPE.EMA, 9, tacuda.MA_TYPE.EMA);
        let count = 0;
        for (let i = 0; i < N; i++) {
            if (Number.isNaN(ref.macd[i]) || Number.isNaN(ext.macd[i])) continue;
            if (Math.abs(ref.macd[i] - ext.macd[i]) > 1e-4) return false;
            count++;
        }
        return count > 0;
    });
    test('MACDEXT(SMA,SMA,SMA) runs', () => {
        const r = tacuda.macdext(close, 12, 0, 26, 0, 9, 0);
        return hasValidValues(r.macd) && hasValidValues(r.signal) && hasValidValues(r.hist);
    });

    // ---- Precision (_ex) ----
    console.log('--- Precision (_ex) ---');
    test('smaEx float64 ~ sma float32', () => {
        const close64 = Float64Array.from(close);
        const out64 = tacuda.smaEx(close64, 20);
        const out32 = tacuda.sma(close, 20);
        for (let i = 0; i < N; i += 53) {
            if (Number.isNaN(out32[i]) !== Number.isNaN(out64[i])) return false;
            if (!Number.isNaN(out32[i]) && Math.abs(out32[i] - out64[i]) > 1e-3) return false;
        }
        return out64 instanceof Float64Array;
    });
    test('rsiEx float64 in range', () => {
        const close64 = Float64Array.from(close);
        const out = tacuda.rsiEx(close64, 14);
        for (let i = 0; i < N; i++) {
            if (!Number.isNaN(out[i]) && (out[i] < -EPS || out[i] > 100 + EPS)) return false;
        }
        return true;
    });

    // ---- Device API ----
    console.log('--- Device API ---');
    test('device pipeline: upload -> sma_d -> download == sma', () => {
        const dIn = tacuda.device.alloc(N);
        const dOut = tacuda.device.alloc(N);
        try {
            tacuda.device.upload(dIn, close);
            tacuda.device.sma(dIn, dOut, N, 20);
            tacuda.device.sync();
            const gpu = tacuda.device.download(dOut, N);
            const cpu = tacuda.sma(close, 20);
            for (let i = 0; i < N; i += 41) {
                if (Number.isNaN(cpu[i]) !== Number.isNaN(gpu[i])) return false;
                if (!Number.isNaN(cpu[i]) && Math.abs(cpu[i] - gpu[i]) > 1e-4) return false;
            }
            return true;
        } finally {
            tacuda.device.free(dIn);
            tacuda.device.free(dOut);
        }
    });
    test('device vector math: sin_d == sin', () => {
        const dIn = tacuda.device.alloc(N);
        const dOut = tacuda.device.alloc(N);
        try {
            tacuda.device.upload(dIn, mathInput);
            tacuda.device.sin(dIn, dOut, N);
            tacuda.device.sync();
            const gpu = tacuda.device.download(dOut, N);
            const host = tacuda.sin(mathInput);
            for (let i = 0; i < N; i += 37) {
                if (Math.abs(gpu[i] - host[i]) > 1e-5) return false;
            }
            return true;
        } finally {
            tacuda.device.free(dIn);
            tacuda.device.free(dOut);
        }
    });
    test('device macdext_d matches host macdext', () => {
        const dIn = tacuda.device.alloc(N);
        const dM = tacuda.device.alloc(N), dS = tacuda.device.alloc(N), dH = tacuda.device.alloc(N);
        try {
            tacuda.device.upload(dIn, close);
            tacuda.device.macdext(dIn, dM, dS, dH, N, 12, 1, 26, 1, 9, 1);
            tacuda.device.sync();
            const gpuMacd = tacuda.device.download(dM, N);
            const host = tacuda.macdext(close, 12, 1, 26, 1, 9, 1);
            for (let i = 0; i < N; i += 43) {
                if (Number.isNaN(host.macd[i]) !== Number.isNaN(gpuMacd[i])) return false;
                if (!Number.isNaN(host.macd[i]) && Math.abs(host.macd[i] - gpuMacd[i]) > 1e-3) return false;
            }
            return true;
        } finally {
            tacuda.device.free(dIn);
            tacuda.device.free(dM);
            tacuda.device.free(dS);
            tacuda.device.free(dH);
        }
    });
    test('bufferPool stats', () => {
        const count = tacuda.bufferPool.allocationCount();
        return Number.isFinite(count) && count >= 0;
    });

    // ---- OHLCV container (structured inputs) ----
    console.log('--- OHLCV container ---');
    const series = tacuda.Ohlcv.fromColumns(open, high, low, close, volume);

    function sameSeries(a, b, tol = 1e-5) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (Number.isNaN(a[i]) !== Number.isNaN(b[i])) return false;
            if (!Number.isNaN(a[i]) && Math.abs(a[i] - b[i]) > tol) return false;
        }
        return true;
    }

    test('atr(Ohlcv) == atr(h,l,c)', () =>
        sameSeries(tacuda.atr(series, 14), tacuda.atr(high, low, close, 14)));
    test('adx(Ohlcv) == adx(h,l,c)', () =>
        sameSeries(tacuda.adx(series, 14), tacuda.adx(high, low, close, 14)));
    test('mfi(Ohlcv) == mfi(h,l,c,v)', () =>
        sameSeries(tacuda.mfi(series, 14), tacuda.mfi(high, low, close, volume, 14)));
    test('obv(Ohlcv) == obv(c,v)', () =>
        sameSeries(tacuda.obv(series), tacuda.obv(close, volume)));
    test('bop(plain object) == bop(o,h,l,c)', () =>
        sameSeries(tacuda.bop({ open, high, low, close }), tacuda.bop(open, high, low, close)));
    test('cdlPattern(doji, Ohlcv) == cdlPattern(doji, o,h,l,c)', () =>
        sameSeries(tacuda.cdlPattern('doji', series), tacuda.cdlPattern('doji', open, high, low, close)));
    test('Ohlcv.fromRows round-trip', () => {
        const rows = [[1, 2, 0.5, 1.5, 100], [1.5, 2.5, 1.0, 2.0, 200]];
        const s = tacuda.Ohlcv.fromRows(rows);
        return s.size === 2 && s.high[1] === 2.5 && s.volume[0] === 100;
    });

    // ---- Batched (multi-symbol) API ----
    console.log('--- Batch API ---');
    const NUM_SERIES = 4, SERIES_LEN = 1024;
    const batchRows = [];
    for (let s = 0; s < NUM_SERIES; s++) {
        const rowsData = generateOHLCV(SERIES_LEN + s * 0);
        batchRows.push(rowsData.close);
    }
    test('rsiBatch matches per-series rsi', () => {
        const batch = tacuda.rsiBatch(batchRows, 14, { maxConcurrency: 3 });
        for (let s = 0; s < NUM_SERIES; s++) {
            if (!sameSeries(batch[s], tacuda.rsi(batchRows[s], 14), 1e-4)) return false;
        }
        return true;
    });
    test('smaBatch (flat matrix input) matches per-series sma', () => {
        const flat = new Float32Array(NUM_SERIES * SERIES_LEN);
        for (let s = 0; s < NUM_SERIES; s++) flat.set(batchRows[s], s * SERIES_LEN);
        const batch = tacuda.smaBatch(flat, 20, { seriesLen: SERIES_LEN });
        for (let s = 0; s < NUM_SERIES; s++) {
            if (!sameSeries(batch[s], tacuda.sma(batchRows[s], 20), 1e-4)) return false;
        }
        return true;
    });
    test('macdBatch matches per-series macd', () => {
        const batch = tacuda.macdBatch(batchRows);
        for (let s = 0; s < NUM_SERIES; s++) {
            const single = tacuda.macd(batchRows[s], 12, 26, 9, 1);
            if (!sameSeries(batch.macd[s], single.macd, 1e-4)) return false;
            if (!sameSeries(batch.signal[s], single.signal, 1e-4)) return false;
        }
        return true;
    });
    test('atrBatch(ohlcvArray) matches per-series atr', () => {
        const ohlcvs = [];
        for (let s = 0; s < NUM_SERIES; s++) {
            const d = generateOHLCV(SERIES_LEN);
            ohlcvs.push(tacuda.Ohlcv.fromColumns(d.open, d.high, d.low, d.close, d.volume));
        }
        const batch = tacuda.atrBatch(ohlcvs, 14, { maxConcurrency: 2 });
        for (let s = 0; s < NUM_SERIES; s++) {
            const single = tacuda.atr(ohlcvs[s].high, ohlcvs[s].low, ohlcvs[s].close, 14);
            if (!sameSeries(batch[s], single, 1e-4)) return false;
        }
        return true;
    });
    test('device.smaBatch matches host smaBatch', () => {
        const flat = new Float32Array(NUM_SERIES * SERIES_LEN);
        for (let s = 0; s < NUM_SERIES; s++) flat.set(batchRows[s], s * SERIES_LEN);
        const hostBatch = tacuda.smaBatch(flat, 20, { seriesLen: SERIES_LEN });
        const dIn = tacuda.device.alloc(flat.length);
        const dOut = tacuda.device.alloc(flat.length);
        try {
            tacuda.device.upload(dIn, flat);
            tacuda.device.smaBatch(dIn, dOut, SERIES_LEN, NUM_SERIES, 20, 2);
            tacuda.device.sync();
            const gpu = tacuda.device.download(dOut, flat.length);
            for (let s = 0; s < NUM_SERIES; s++) {
                const rowGpu = gpu.subarray(s * SERIES_LEN, (s + 1) * SERIES_LEN);
                if (!sameSeries(rowGpu, hostBatch[s], 1e-4)) return false;
            }
            return true;
        } finally {
            tacuda.device.free(dIn);
            tacuda.device.free(dOut);
        }
    });

    // ---- Generated batch families (0.3.0) ----
    console.log('--- Generated batch families ---');
    const batchHighs = batchRows.map(r => Float32Array.from(r, v => v + 1.0));
    const batchLows = batchRows.map(r => Float32Array.from(r, v => v - 1.0));
    const batchVolumes = batchRows.map(r =>
        Float32Array.from(r, (_, i) => 1000 + 500 * Math.abs(Math.sin(i))));
    test('wmaBatch matches per-series wma', () => {
        const batch = tacuda.wmaBatch(batchRows, 10, { maxConcurrency: 2 });
        for (let s = 0; s < NUM_SERIES; s++) {
            if (!sameSeries(batch[s], tacuda.wma(batchRows[s], 10), 1e-4)) return false;
        }
        return true;
    });
    test('adxBatch matches per-series adx', () => {
        const batch = tacuda.adxBatch(batchHighs, batchLows, batchRows, 14);
        for (let s = 0; s < NUM_SERIES; s++) {
            const single = tacuda.adx(batchHighs[s], batchLows[s], batchRows[s], 14);
            if (!sameSeries(batch[s], single, 1e-3)) return false;
        }
        return true;
    });
    test('mfiBatch matches per-series mfi', () => {
        const batch = tacuda.mfiBatch(batchHighs, batchLows, batchRows, batchVolumes, 14);
        for (let s = 0; s < NUM_SERIES; s++) {
            const single = tacuda.mfi(batchHighs[s], batchLows[s], batchRows[s], batchVolumes[s], 14);
            if (!sameSeries(batch[s], single, 1e-3)) return false;
        }
        return true;
    });
    test('macdextBatch returns macd/signal/hist per series', () => {
        const batch = tacuda.macdextBatch(batchRows, 12, 1, 26, 1, 9, 1);
        const single = tacuda.macd(batchRows[0], 12, 26, 9, 1);
        return sameSeries(batch.macd[0], single.macd, 1e-4)
            && sameSeries(batch.signal[0], single.signal, 1e-4);
    });
    test('aroonBatch returns bounded down/up rows', () => {
        const batch = tacuda.aroonBatch(batchHighs, batchLows, 14);
        if (batch.down.length !== NUM_SERIES || batch.up.length !== NUM_SERIES) return false;
        for (const rows of [batch.down, batch.up]) {
            for (const r of rows) {
                for (const v of r) {
                    if (!Number.isNaN(v) && (v < 0 || v > 100)) return false;
                }
            }
        }
        return true;
    });
    test('trangeBatch (no scalars) matches per-series trange', () => {
        const batch = tacuda.trangeBatch(batchHighs, batchLows, batchRows);
        for (let s = 0; s < NUM_SERIES; s++) {
            const single = tacuda.trange(batchHighs[s], batchLows[s], batchRows[s]);
            if (!sameSeries(batch[s], single, 1e-4)) return false;
        }
        return true;
    });
    test('batch rejects same-element-count but different-shape inputs', () => {
        // 2x10 vs 4x5: identical element count, different shape. Both the
        // hand-written and the generated multi-input wrappers must reject it.
        const rows2x10 = [new Float32Array(10).fill(1), new Float32Array(10).fill(2)];
        const rows4x5 = [1, 2, 3, 4].map(v => new Float32Array(5).fill(v));
        const throws = (fn) => {
            try { fn(); return false; } catch (e) { return e instanceof RangeError; }
        };
        return throws(() => tacuda.atrBatch(rows2x10, rows4x5, rows2x10, 3))
            && throws(() => tacuda.stochasticBatch(rows2x10, rows4x5, rows2x10, 3, 2))
            && throws(() => tacuda.adxBatch(rows2x10, rows4x5, rows2x10, 3))
            && throws(() => tacuda.mfiBatch(rows2x10, rows4x5, rows2x10, rows2x10, 3));
    });
    test('adxBatch(ohlcvArray) accepts Ohlcv containers', () => {
        const ohlcvs = [];
        for (let s = 0; s < NUM_SERIES; s++) {
            ohlcvs.push(tacuda.Ohlcv.fromColumns(
                batchRows[s], batchHighs[s], batchLows[s], batchRows[s], batchVolumes[s]));
        }
        const batch = tacuda.adxBatch(ohlcvs, 14);
        const single = tacuda.adx(batchHighs[0], batchLows[0], batchRows[0], 14);
        return sameSeries(batch[0], single, 1e-3);
    });
    test('device.rocBatch matches host rocBatch', () => {
        const flat = new Float32Array(NUM_SERIES * SERIES_LEN);
        for (let s = 0; s < NUM_SERIES; s++) flat.set(batchRows[s], s * SERIES_LEN);
        const hostBatch = tacuda.rocBatch(flat, 10, { seriesLen: SERIES_LEN });
        const dIn = tacuda.device.alloc(flat.length);
        const dOut = tacuda.device.alloc(flat.length);
        try {
            tacuda.device.upload(dIn, flat);
            tacuda.device.rocBatch(dIn, dOut, SERIES_LEN, NUM_SERIES, 10, 2);
            tacuda.device.sync();
            const gpu = tacuda.device.download(dOut, flat.length);
            for (let s = 0; s < NUM_SERIES; s++) {
                const rowGpu = gpu.subarray(s * SERIES_LEN, (s + 1) * SERIES_LEN);
                if (!sameSeries(rowGpu, hostBatch[s], 1e-4)) return false;
            }
            return true;
        } finally {
            tacuda.device.free(dIn);
            tacuda.device.free(dOut);
        }
    });

    // ---- Metadata ----
    console.log('--- Metadata ---');
    test('version() returns semver string', () => {
        const v = tacuda.version();
        return typeof v === 'string' && v.split('.').length === 3;
    });

    // ---- Cross-validation ----
    console.log('--- Cross-validation ---');
    test('SMA(20) == BBands.Middle', () => {
        const smaOut = tacuda.sma(close, 20);
        const bb = tacuda.bbands(close, 20, 2.0, 2.0);
        let maxDiff = 0, count = 0;
        for (let i = 0; i < N; i++) {
            if (!Number.isNaN(smaOut[i]) && !Number.isNaN(bb.middle[i])) {
                maxDiff = Math.max(maxDiff, Math.abs(smaOut[i] - bb.middle[i]));
                count++;
            }
        }
        return count > 0 && maxDiff < 1e-4;
    });

    // ---- Streaming API ----
    console.log();
    console.log('--- Streaming (incremental update) ---');

    let streamed = null;
    test('RsiStream reproduces the full-series result', () => {
        const s = new tacuda.RsiStream(14);
        try {
            if (s.lookback !== 14 || s.outputs !== 1) return false;
            const half = N >> 1;
            const warm = s.append(close.subarray(0, half));
            const out = new Float32Array(N);
            out.set(warm, 0);
            for (let i = half; i < N; ++i) out[i] = s.append(close.subarray(i, i + 1))[0];
            if (s.bars !== N) return false;
            streamed = out;
        } finally {
            s.destroy();
        }
        const full = tacuda.rsi(close, 14);
        let maxDiff = 0;
        for (let i = 0; i < N; ++i) {
            const a = streamed[i], b = full[i];
            if (Number.isNaN(a) !== Number.isNaN(b)) return false;
            if (!Number.isNaN(a)) maxDiff = Math.max(maxDiff, Math.abs(a - b));
        }
        return maxDiff < 1e-3;
    });

    test('RsiStream bulk append == bar-by-bar', () => {
        const s = new tacuda.RsiStream(14);
        try {
            const bulk = s.append(close);
            for (let i = 0; i < N; ++i) {
                const a = bulk[i], b = streamed[i];
                if (Number.isNaN(a) !== Number.isNaN(b)) return false;
                if (!Number.isNaN(a) && a !== b) return false;
            }
            return true;
        } finally {
            s.destroy();
        }
    });

    test('MacdStream returns three lines', () => {
        const s = new tacuda.MacdStream(12, 26, 9);
        try {
            const lines = s.append(close);
            return Array.isArray(lines) && lines.length === 3 && s.lookback === 33;
        } finally {
            s.destroy();
        }
    });

    test('RsiStreamBatch(GPU) rows == single stream', () => {
        const series = 4;
        const b = new tacuda.RsiStreamBatch(14, series);
        try {
            b.setExec(tacuda.stream.EXEC_GPU);
            const mat = new Float32Array(series * N);
            for (let r = 0; r < series; ++r) mat.set(close, r * N);
            const out = b.append(mat);
            if (b.numSeries !== series) return false;
            for (let r = 0; r < series; ++r) {
                for (let i = 0; i < N; ++i) {
                    const a = out[r * N + i], c = streamed[i];
                    if (Number.isNaN(a) !== Number.isNaN(c)) return false;
                    if (!Number.isNaN(a) && a !== c) return false;
                }
            }
            return true;
        } finally {
            b.destroy();
        }
    });

    test('destroyed stream handle is rejected', () => {
        const s = new tacuda.SmaStream(10);
        s.destroy();
        try {
            s.append(close.subarray(0, 1));
            return false;
        } catch (e) {
            return true;
        }
    });

    test('stream append rejects mismatched input lengths', () => {
        const s = new tacuda.AtrStream(14);
        try {
            s.append(high, low.subarray(0, N - 1), close);
            return false;
        } catch (e) {
            return e instanceof RangeError;
        } finally {
            s.destroy();
        }
    });

    // ---- Results ----
    console.log();
    console.log('='.repeat(70));
    const total = results.passed + results.failed;
    console.log(`RESULTS: ${results.passed} passed, ${results.failed} failed, ${total} total`);
    console.log('='.repeat(70));

    return results.failed > 0 ? 1 : 0;
}

if (require.main === module) {
    process.exit(runAllTests());
}

module.exports = { runAllTests };
