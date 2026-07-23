// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
/**
 * TACUDA Node.js Bindings
 * 
 * Koffi-based bindings for the TACUDA GPU-accelerated technical analysis library.
 * Covers ALL host-API indicators from tacuda.h.
 */

const koffi = require('koffi');
const path = require('path');
const fs = require('fs');

// ============================================================================
// Library Loading
// ============================================================================

function findLibrary() {
    const candidates = [];
    
    if (process.env.TACUDA_LIBRARY) {
        candidates.push(process.env.TACUDA_LIBRARY);
    }
    
    const baseDir = path.resolve(__dirname, '..', '..');
    
    if (process.platform === 'win32') {
        candidates.push(
            path.join(baseDir, 'build', 'Release', 'tacuda.dll'),
            path.join(baseDir, 'build', 'Debug', 'tacuda.dll'),
            path.join(baseDir, 'Release', 'tacuda.dll'),
            path.join(__dirname, 'tacuda.dll'),
            'tacuda.dll'
        );
    } else if (process.platform === 'darwin') {
        candidates.push(
            path.join(baseDir, 'build', 'libtacuda.dylib'),
            path.join(baseDir, 'lib', 'libtacuda.dylib'),
            path.join(__dirname, 'libtacuda.dylib'),
            'libtacuda.dylib'
        );
    } else {
        candidates.push(
            path.join(baseDir, 'build', 'libtacuda.so'),
            path.join(baseDir, 'lib', 'libtacuda.so'),
            path.join(__dirname, 'libtacuda.so'),
            'libtacuda.so'
        );
    }
    
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    
    throw new Error(
        'TACUDA library not found. Build the project first or set TACUDA_LIBRARY environment variable.\n' +
        'Searched in: ' + candidates.slice(0, 5).join(', ')
    );
}

const libPath = findLibrary();
const lib = koffi.load(libPath);

// ============================================================================
// FFI Declarations  (cudaStream_t = void*, always pass null = stream 0)
// ============================================================================

// --- Moving Averages ---
const ct_sma   = lib.func('int ct_sma(const float *input, float *output, int size, int period, void *stream)');
const ct_ma    = lib.func('int ct_ma(const float *input, float *output, int size, int period, int type, void *stream)');
const ct_mavp  = lib.func('int ct_mavp(const float *input, const float *periods, float *output, int size, int minPeriod, int maxPeriod, int type, void *stream)');
const ct_wma   = lib.func('int ct_wma(const float *input, float *output, int size, int period, void *stream)');
const ct_ema   = lib.func('int ct_ema(const float *input, float *output, int size, int period, void *stream)');
const ct_dema  = lib.func('int ct_dema(const float *input, float *output, int size, int period, void *stream)');
const ct_tema  = lib.func('int ct_tema(const float *input, float *output, int size, int period, void *stream)');
const ct_t3    = lib.func('int ct_t3(const float *input, float *output, int size, int period, float vFactor, void *stream)');
const ct_trima = lib.func('int ct_trima(const float *input, float *output, int size, int period, void *stream)');
const ct_kama  = lib.func('int ct_kama(const float *input, float *output, int size, int period, int fastPeriod, int slowPeriod, void *stream)');

// --- Momentum ---
const ct_momentum = lib.func('int ct_momentum(const float *input, float *output, int size, int period, void *stream)');
const ct_change   = lib.func('int ct_change(const float *input, float *output, int size, int period, void *stream)');
const ct_roc      = lib.func('int ct_roc(const float *input, float *output, int size, int period, void *stream)');
const ct_rocp     = lib.func('int ct_rocp(const float *input, float *output, int size, int period, void *stream)');
const ct_rocr     = lib.func('int ct_rocr(const float *input, float *output, int size, int period, void *stream)');
const ct_rocr100  = lib.func('int ct_rocr100(const float *input, float *output, int size, int period, void *stream)');
const ct_trix     = lib.func('int ct_trix(const float *input, float *output, int size, int period, void *stream)');
const ct_cmo      = lib.func('int ct_cmo(const float *input, float *output, int size, int period, void *stream)');
const ct_rsi      = lib.func('int ct_rsi(const float *input, float *output, int size, int period, void *stream)');
const ct_apo      = lib.func('int ct_apo(const float *input, float *output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_ppo      = lib.func('int ct_ppo(const float *input, float *output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_apo_ma   = lib.func('int ct_apo_ma(const float *input, float *output, int size, int fastPeriod, int slowPeriod, int type, void *stream)');
const ct_ppo_ma   = lib.func('int ct_ppo_ma(const float *input, float *output, int size, int fastPeriod, int slowPeriod, int type, void *stream)');
const ct_pvo      = lib.func('int ct_pvo(const float *volume, float *output, int size, int fastPeriod, int slowPeriod, void *stream)');

// --- MACD family ---
const ct_macd_line = lib.func('int ct_macd_line(const float *input, float *output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_macd      = lib.func('int ct_macd(const float *input, float *macd, float *signal, float *hist, int size, int fastPeriod, int slowPeriod, int signalPeriod, int maType, void *stream)');
const ct_macdfix   = lib.func('int ct_macdfix(const float *input, float *macd, float *signal, float *hist, int size, int signalPeriod, void *stream)');

// --- MAMA ---
const ct_mama = lib.func('int ct_mama(const float *input, float *mama, float *fama, int size, float fastLimit, float slowLimit, void *stream)');

// --- Statistics ---
const ct_stddev  = lib.func('int ct_stddev(const float *input, float *output, int size, int period, void *stream)');
const ct_avgdev  = lib.func('int ct_avgdev(const float *input, float *output, int size, int period, void *stream)');
const ct_var     = lib.func('int ct_var(const float *input, float *output, int size, int period, void *stream)');
const ct_sum     = lib.func('int ct_sum(const float *input, float *output, int size, int period, void *stream)');
const ct_max     = lib.func('int ct_max(const float *input, float *output, int size, int period, void *stream)');
const ct_min     = lib.func('int ct_min(const float *input, float *output, int size, int period, void *stream)');
const ct_maxindex = lib.func('int ct_maxindex(const float *input, float *output, int size, int period, void *stream)');
const ct_minindex = lib.func('int ct_minindex(const float *input, float *output, int size, int period, void *stream)');
const ct_minmax      = lib.func('int ct_minmax(const float *input, float *min, float *max, int size, int period, void *stream)');
const ct_minmaxindex = lib.func('int ct_minmaxindex(const float *input, float *minidx, float *maxidx, int size, int period, void *stream)');
const ct_beta   = lib.func('int ct_beta(const float *x, const float *y, float *output, int size, int period, void *stream)');
const ct_correl = lib.func('int ct_correl(const float *x, const float *y, float *output, int size, int period, void *stream)');

// --- Linear Regression family ---
const ct_linearreg           = lib.func('int ct_linearreg(const float *input, float *output, int size, int period, void *stream)');
const ct_linearreg_slope     = lib.func('int ct_linearreg_slope(const float *input, float *output, int size, int period, void *stream)');
const ct_linearreg_intercept = lib.func('int ct_linearreg_intercept(const float *input, float *output, int size, int period, void *stream)');
const ct_linearreg_angle     = lib.func('int ct_linearreg_angle(const float *input, float *output, int size, int period, void *stream)');
const ct_tsf                 = lib.func('int ct_tsf(const float *input, float *output, int size, int period, void *stream)');
const ct_midpoint            = lib.func('int ct_midpoint(const float *input, float *output, int size, int period, void *stream)');

// --- Volatility ---
const ct_bbands  = lib.func('int ct_bbands(const float *input, float *upper, float *middle, float *lower, int size, int period, float upperMul, float lowerMul, void *stream)');
const ct_accbands = lib.func('int ct_accbands(const float *high, const float *low, const float *close, float *upper, float *middle, float *lower, int size, int period, void *stream)');
const ct_atr     = lib.func('int ct_atr(const float *high, const float *low, const float *close, float *output, int size, int period, float initial, void *stream)');
const ct_natr    = lib.func('int ct_natr(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_trange  = lib.func('int ct_trange(const float *high, const float *low, const float *close, float *output, int size, void *stream)');

// --- Oscillators ---
const ct_stochastic = lib.func('int ct_stochastic(const float *high, const float *low, const float *close, float *k, float *d, int size, int kPeriod, int dPeriod, void *stream)');
const ct_stochf     = lib.func('int ct_stochf(const float *high, const float *low, const float *close, float *k, float *d, int size, int kPeriod, int dPeriod, void *stream)');
const ct_stochrsi   = lib.func('int ct_stochrsi(const float *input, float *k, float *d, int size, int rsiPeriod, int kPeriod, int dPeriod, void *stream)');
const ct_cci   = lib.func('int ct_cci(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_willr = lib.func('int ct_willr(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_ultosc = lib.func('int ct_ultosc(const float *high, const float *low, const float *close, float *output, int size, int shortPeriod, int mediumPeriod, int longPeriod, void *stream)');

// --- Directional Movement ---
const ct_adx      = lib.func('int ct_adx(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_adxr     = lib.func('int ct_adxr(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_dx       = lib.func('int ct_dx(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_plus_dm  = lib.func('int ct_plus_dm(const float *high, const float *low, float *output, int size, int period, void *stream)');
const ct_minus_dm = lib.func('int ct_minus_dm(const float *high, const float *low, float *output, int size, int period, void *stream)');
const ct_plus_di  = lib.func('int ct_plus_di(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');
const ct_minus_di = lib.func('int ct_minus_di(const float *high, const float *low, const float *close, float *output, int size, int period, void *stream)');

// --- Volume ---
const ct_obv  = lib.func('int ct_obv(const float *close, const float *volume, float *output, int size, void *stream)');
const ct_mfi  = lib.func('int ct_mfi(const float *high, const float *low, const float *close, const float *volume, float *output, int size, int period, void *stream)');
const ct_ad   = lib.func('int ct_ad(const float *high, const float *low, const float *close, const float *volume, float *output, int size, void *stream)');
const ct_adosc = lib.func('int ct_adosc(const float *high, const float *low, const float *close, const float *volume, float *output, int size, int shortPeriod, int longPeriod, void *stream)');
const ct_imi  = lib.func('int ct_imi(const float *open, const float *close, float *output, int size, int period, void *stream)');
const ct_nvi  = lib.func('int ct_nvi(const float *close, const float *volume, float *output, int size, void *stream)');
const ct_pvi  = lib.func('int ct_pvi(const float *close, const float *volume, float *output, int size, void *stream)');

// --- SAR ---
const ct_sar    = lib.func('int ct_sar(const float *high, const float *low, float *output, int size, float step, float maxAcceleration, void *stream)');
const ct_sarext = lib.func('int ct_sarext(const float *high, const float *low, float *output, int size, float startValue, float offsetOnReverse, float accInitLong, float accLong, float accMaxLong, float accInitShort, float accShort, float accMaxShort, void *stream)');

// --- MACDEXT (independent MA types per line, TA-Lib MACDEXT) ---
const ct_macdext = lib.func('int ct_macdext(const float *input, float *macd, float *signal, float *hist, int size, int fastPeriod, int fastType, int slowPeriod, int slowType, int signalPeriod, int signalType, void *stream)');

// --- Vector Math (element-wise, no lookback) ---
const UNARY_MATH_NAMES = [
    'acos', 'asin', 'atan', 'ceil', 'cos', 'cosh', 'exp', 'floor',
    'ln', 'log10', 'sin', 'sinh', 'sqrt', 'tan', 'tanh',
];
const BINARY_MATH_NAMES = ['add', 'sub', 'mult', 'div'];

const unaryMathFuncs = {};
for (const name of UNARY_MATH_NAMES) {
    unaryMathFuncs[name] = lib.func(
        `int ct_${name}(const float *input, float *output, int size, void *stream)`
    );
}
const binaryMathFuncs = {};
for (const name of BINARY_MATH_NAMES) {
    binaryMathFuncs[name] = lib.func(
        `int ct_${name}(const float *input1, const float *input2, float *output, int size, void *stream)`
    );
}

// --- Precision (_ex) host functions: storage/compute dtype selection ---
const ct_sma_ex    = lib.func('int ct_sma_ex(const void *input, void *output, int size, int period, int storageType, int computeType, void *stream)');
const ct_ema_ex    = lib.func('int ct_ema_ex(const void *input, void *output, int size, int period, int storageType, int computeType, void *stream)');
const ct_rsi_ex    = lib.func('int ct_rsi_ex(const void *input, void *output, int size, int period, int storageType, int computeType, void *stream)');
const ct_macd_ex   = lib.func('int ct_macd_ex(const void *input, void *macd, void *signal, void *hist, int size, int fastPeriod, int slowPeriod, int signalPeriod, int maType, int storageType, int computeType, void *stream)');
const ct_atr_ex    = lib.func('int ct_atr_ex(const void *high, const void *low, const void *close, void *output, int size, int period, float initial, int storageType, int computeType, void *stream)');
const ct_bbands_ex = lib.func('int ct_bbands_ex(const void *input, void *upper, void *middle, void *lower, int size, int period, float upperMul, float lowerMul, int storageType, int computeType, void *stream)');

// --- Device memory management ---
const ct_device_alloc    = lib.func('void* ct_device_alloc(int count)');
const ct_device_alloc_ex = lib.func('void* ct_device_alloc_ex(int count, int storageType)');
const ct_device_free     = lib.func('void ct_device_free(void *d_ptr)');
const ct_device_free_ex  = lib.func('void ct_device_free_ex(void *d_ptr)');
const ct_device_upload   = lib.func('int ct_device_upload(void *d_dst, const float *h_src, int count, void *stream)');
const ct_device_download = lib.func('int ct_device_download(float *h_dst, const void *d_src, int count, void *stream)');
const ct_device_upload_ex   = lib.func('int ct_device_upload_ex(void *d_dst, const void *h_src, int count, int storageType, void *stream)');
const ct_device_download_ex = lib.func('int ct_device_download_ex(void *h_dst, const void *d_src, int count, int storageType, void *stream)');
const ct_device_sync     = lib.func('int ct_device_sync(void *stream)');

// --- DeviceBufferPool management ---
const ct_device_buffer_pool_cleanup = lib.func('void ct_device_buffer_pool_cleanup()');
const ct_device_buffer_pool_allocation_count = lib.func('uint64_t ct_device_buffer_pool_allocation_count()');
const ct_device_buffer_pool_set_max_cache_bytes = lib.func('uint64_t ct_device_buffer_pool_set_max_cache_bytes(uint64_t bytes)');

// --- Library metadata ---
const ct_version = lib.func('const char* ct_version()');

// --- Batched (multi-symbol) API: series-major matrices numSeries x seriesLen ---
const ct_sma_batch = lib.func('int ct_sma_batch(const float *inputs, float *outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_ema_batch = lib.func('int ct_ema_batch(const float *inputs, float *outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_rsi_batch = lib.func('int ct_rsi_batch(const float *inputs, float *outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_atr_batch = lib.func('int ct_atr_batch(const float *highs, const float *lows, const float *closes, float *outputs, int seriesLen, int numSeries, int period, float initial, int maxConcurrency, void *stream)');
const ct_macd_batch = lib.func('int ct_macd_batch(const float *inputs, float *macds, float *signals, float *hists, int seriesLen, int numSeries, int fastPeriod, int slowPeriod, int signalPeriod, int maType, int maxConcurrency, void *stream)');
const ct_bbands_batch = lib.func('int ct_bbands_batch(const float *inputs, float *uppers, float *middles, float *lowers, int seriesLen, int numSeries, int period, float upperMul, float lowerMul, int maxConcurrency, void *stream)');
const ct_stochastic_batch = lib.func('int ct_stochastic_batch(const float *highs, const float *lows, const float *closes, float *ks, float *ds, int seriesLen, int numSeries, int kPeriod, int dPeriod, int maxConcurrency, void *stream)');
// Device-pointer batch variants
const ct_sma_batch_d = lib.func('int ct_sma_batch_d(const void *d_inputs, void *d_outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_ema_batch_d = lib.func('int ct_ema_batch_d(const void *d_inputs, void *d_outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_rsi_batch_d = lib.func('int ct_rsi_batch_d(const void *d_inputs, void *d_outputs, int seriesLen, int numSeries, int period, int maxConcurrency, void *stream)');
const ct_atr_batch_d = lib.func('int ct_atr_batch_d(const void *d_highs, const void *d_lows, const void *d_closes, void *d_outputs, int seriesLen, int numSeries, int period, float initial, int maxConcurrency, void *stream)');
const ct_macd_batch_d = lib.func('int ct_macd_batch_d(const void *d_inputs, void *d_macds, void *d_signals, void *d_hists, int seriesLen, int numSeries, int fastPeriod, int slowPeriod, int signalPeriod, int maType, int maxConcurrency, void *stream)');
const ct_bbands_batch_d = lib.func('int ct_bbands_batch_d(const void *d_inputs, void *d_uppers, void *d_middles, void *d_lowers, int seriesLen, int numSeries, int period, float upperMul, float lowerMul, int maxConcurrency, void *stream)');
const ct_stochastic_batch_d = lib.func('int ct_stochastic_batch_d(const void *d_highs, const void *d_lows, const void *d_closes, void *d_ks, void *d_ds, int seriesLen, int numSeries, int kPeriod, int dPeriod, int maxConcurrency, void *stream)');

// --- Device API (_d suffix, raw GPU pointers) ---
// Single input -> single output with period: (d_in, d_out, size, period)
const DEVICE_PERIOD_FUNCS = [
    'sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'rsi', 'momentum',
    'roc', 'rocp', 'rocr', 'rocr100', 'cmo', 'stddev', 'var', 'sum',
    'max', 'min', 'linearreg', 'linearreg_slope', 'linearreg_intercept',
    'linearreg_angle', 'tsf',
];
const devicePeriodFuncs = {};
for (const name of DEVICE_PERIOD_FUNCS) {
    devicePeriodFuncs[name] = lib.func(
        `int ct_${name}_d(const void *d_input, void *d_output, int size, int period, void *stream)`
    );
}
// Vector math on device pointers
const deviceUnaryMathFuncs = {};
for (const name of UNARY_MATH_NAMES) {
    deviceUnaryMathFuncs[name] = lib.func(
        `int ct_${name}_d(const void *d_input, void *d_output, int size, void *stream)`
    );
}
const deviceBinaryMathFuncs = {};
for (const name of BINARY_MATH_NAMES) {
    deviceBinaryMathFuncs[name] = lib.func(
        `int ct_${name}_d(const void *d_input1, const void *d_input2, void *d_output, int size, void *stream)`
    );
}
// HLC input indicators: (d_high, d_low, d_close, d_out, size, period)
const DEVICE_HLC_PERIOD_FUNCS = ['natr', 'adx', 'adxr', 'dx', 'plus_di', 'minus_di', 'willr', 'cci'];
const deviceHlcPeriodFuncs = {};
for (const name of DEVICE_HLC_PERIOD_FUNCS) {
    deviceHlcPeriodFuncs[name] = lib.func(
        `int ct_${name}_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, int period, void *stream)`
    );
}
const ct_kama_d   = lib.func('int ct_kama_d(const void *d_input, void *d_output, int size, int period, int fastPeriod, int slowPeriod, void *stream)');
const ct_t3_d     = lib.func('int ct_t3_d(const void *d_input, void *d_output, int size, int period, float vFactor, void *stream)');
const ct_atr_d    = lib.func('int ct_atr_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, int period, float percentile, void *stream)');
const ct_trange_d = lib.func('int ct_trange_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, void *stream)');
const ct_macd_d   = lib.func('int ct_macd_d(const void *d_input, void *d_macd, void *d_signal, void *d_hist, int size, int fastPeriod, int slowPeriod, int signalPeriod, int maType, void *stream)');
const ct_macdext_d = lib.func('int ct_macdext_d(const void *d_input, void *d_macd, void *d_signal, void *d_hist, int size, int fastPeriod, int fastType, int slowPeriod, int slowType, int signalPeriod, int signalType, void *stream)');
const ct_stochastic_d = lib.func('int ct_stochastic_d(const void *d_high, const void *d_low, const void *d_close, void *d_slowK, void *d_slowD, int size, int kPeriod, int dPeriod, void *stream)');
const ct_stochastic_fast_d = lib.func('int ct_stochastic_fast_d(const void *d_high, const void *d_low, const void *d_close, void *d_fastK, void *d_fastD, int size, int kPeriod, int dPeriod, int dMaType, void *stream)');
const ct_ultosc_d = lib.func('int ct_ultosc_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, int period1, int period2, int period3, void *stream)');
const ct_bbands_d = lib.func('int ct_bbands_d(const void *d_input, void *d_upper, void *d_middle, void *d_lower, int size, int period, float nbDevUp, float nbDevDn, void *stream)');
const ct_accbands_d = lib.func('int ct_accbands_d(const void *d_high, const void *d_low, const void *d_close, void *d_upper, void *d_middle, void *d_lower, int size, int period, void *stream)');
const ct_ad_d     = lib.func('int ct_ad_d(const void *d_high, const void *d_low, const void *d_close, const void *d_volume, void *d_output, int size, void *stream)');
const ct_adosc_d  = lib.func('int ct_adosc_d(const void *d_high, const void *d_low, const void *d_close, const void *d_volume, void *d_output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_obv_d    = lib.func('int ct_obv_d(const void *d_close, const void *d_volume, void *d_output, int size, void *stream)');
const ct_mfi_d    = lib.func('int ct_mfi_d(const void *d_high, const void *d_low, const void *d_close, const void *d_volume, void *d_output, int size, int period, void *stream)');
const ct_avgprice_d = lib.func('int ct_avgprice_d(const void *d_open, const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, void *stream)');
const ct_medprice_d = lib.func('int ct_medprice_d(const void *d_high, const void *d_low, void *d_output, int size, void *stream)');
const ct_typprice_d = lib.func('int ct_typprice_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, void *stream)');
const ct_wclprice_d = lib.func('int ct_wclprice_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, void *stream)');
const ct_aroon_d    = lib.func('int ct_aroon_d(const void *d_high, const void *d_low, void *d_aroonDown, void *d_aroonUp, int size, int period, void *stream)');
const ct_aroonosc_d = lib.func('int ct_aroonosc_d(const void *d_high, const void *d_low, void *d_output, int size, int period, void *stream)');
// Precision (_ex_d) device functions
const ct_sma_ex_d    = lib.func('int ct_sma_ex_d(const void *d_input, void *d_output, int size, int period, int storageType, int computeType, void *stream)');
const ct_ema_ex_d    = lib.func('int ct_ema_ex_d(const void *d_input, void *d_output, int size, int period, int storageType, int computeType, void *stream)');
const ct_rsi_ex_d    = lib.func('int ct_rsi_ex_d(const void *d_input, void *d_output, int size, int period, int storageType, int computeType, void *stream)');
const ct_atr_ex_d    = lib.func('int ct_atr_ex_d(const void *d_high, const void *d_low, const void *d_close, void *d_output, int size, int period, float percentile, int storageType, int computeType, void *stream)');
const ct_macd_ex_d   = lib.func('int ct_macd_ex_d(const void *d_input, void *d_macd, void *d_signal, void *d_hist, int size, int fastPeriod, int slowPeriod, int signalPeriod, int maType, int storageType, int computeType, void *stream)');
const ct_bbands_ex_d = lib.func('int ct_bbands_ex_d(const void *d_input, void *d_upper, void *d_middle, void *d_lower, int size, int period, float nbDevUp, float nbDevDn, int storageType, int computeType, void *stream)');

// Legacy ct_*_device host-header functions (device pointers, same semantics as _d)
const DEVICE_LEGACY_PERIOD_FUNCS = [
    'sma', 'wma', 'momentum', 'change', 'roc', 'rocp', 'rocr', 'rocr100',
    'ema', 'dema', 'tema', 'trima', 'trix', 'max', 'min', 'maxindex',
    'minindex', 'stddev', 'var', 'sum', 'rsi',
];
const deviceLegacyPeriodFuncs = {};
for (const name of DEVICE_LEGACY_PERIOD_FUNCS) {
    deviceLegacyPeriodFuncs[name] = lib.func(
        `int ct_${name}_device(const void *d_input, void *d_output, int size, int period, void *stream)`
    );
}
const ct_ma_device        = lib.func('int ct_ma_device(const void *d_input, void *d_output, int size, int period, int type, void *stream)');
const ct_kama_device      = lib.func('int ct_kama_device(const void *d_input, void *d_output, int size, int period, int fastPeriod, int slowPeriod, void *stream)');
const ct_t3_device        = lib.func('int ct_t3_device(const void *d_input, void *d_output, int size, int period, float vFactor, void *stream)');
const ct_macd_line_device = lib.func('int ct_macd_line_device(const void *d_input, void *d_output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_apo_device       = lib.func('int ct_apo_device(const void *d_input, void *d_output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_ppo_device       = lib.func('int ct_ppo_device(const void *d_input, void *d_output, int size, int fastPeriod, int slowPeriod, void *stream)');
const ct_pvo_device       = lib.func('int ct_pvo_device(const void *d_volume, void *d_output, int size, int fastPeriod, int slowPeriod, void *stream)');

// --- Aroon ---
const ct_aroon    = lib.func('int ct_aroon(const float *high, const float *low, float *up, float *down, float *osc, int size, int upPeriod, int downPeriod, void *stream)');
const ct_aroonosc = lib.func('int ct_aroonosc(const float *high, const float *low, float *output, int size, int period, void *stream)');

// --- Price Transform ---
const ct_avgprice = lib.func('int ct_avgprice(const float *open, const float *high, const float *low, const float *close, float *output, int size, void *stream)');
const ct_medprice = lib.func('int ct_medprice(const float *high, const float *low, float *output, int size, void *stream)');
const ct_typprice = lib.func('int ct_typprice(const float *high, const float *low, const float *close, float *output, int size, void *stream)');
const ct_wclprice = lib.func('int ct_wclprice(const float *high, const float *low, const float *close, float *output, int size, void *stream)');
const ct_midprice = lib.func('int ct_midprice(const float *high, const float *low, float *output, int size, int period, void *stream)');
const ct_bop      = lib.func('int ct_bop(const float *open, const float *high, const float *low, const float *close, float *output, int size, void *stream)');

// --- Hilbert Transform ---
const ct_ht_dcperiod  = lib.func('int ct_ht_dcperiod(const float *input, float *output, int size, void *stream)');
const ct_ht_dcphase   = lib.func('int ct_ht_dcphase(const float *input, float *output, int size, void *stream)');
const ct_ht_phasor    = lib.func('int ct_ht_phasor(const float *input, float *inphase, float *quadrature, int size, void *stream)');
const ct_ht_sine      = lib.func('int ct_ht_sine(const float *input, float *sine, float *leadsine, int size, void *stream)');
const ct_ht_trendline = lib.func('int ct_ht_trendline(const float *input, float *output, int size, void *stream)');
const ct_ht_trendmode = lib.func('int ct_ht_trendmode(const float *input, float *output, int size, void *stream)');

// --- Candlestick Patterns (all share the same signature) ---
const CDL_NAMES = [
    'doji', 'hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing',
    'three_white_soldiers', 'abandoned_baby', 'advance_block', 'belt_hold', 'breakaway',
    'two_crows', 'three_black_crows', 'three_inside', 'three_outside', 'three_line_strike',
    'three_stars_in_south', 'closing_marubozu', 'conceal_baby_swallow', 'counterattack',
    'dark_cloud_cover', 'doji_star', 'dragonfly_doji', 'engulfing', 'evening_doji_star',
    'evening_star', 'gap_side_side_white', 'gravestone_doji', 'hanging_man', 'harami',
    'harami_cross', 'high_wave', 'hikkake', 'hikkake_mod', 'homing_pigeon',
    'identical_three_crows', 'in_neck', 'kicking', 'kicking_by_length', 'ladder_bottom',
    'long_legged_doji', 'long_line', 'marubozu', 'matching_low', 'mat_hold',
    'morning_doji_star', 'morning_star', 'on_neck', 'piercing', 'rickshaw_man',
    'rise_fall3_methods', 'separating_lines', 'shooting_star', 'short_line', 'spinning_top',
    'stalled_pattern', 'stick_sandwich', 'takuri', 'tasuki_gap', 'thrusting',
    'tristar', 'unique_3_river', 'upside_gap_2_crows', 'xside_gap_3_methods',
];

const cdlFuncs = {};
for (const name of CDL_NAMES) {
    cdlFuncs[name] = lib.func(
        `int ct_cdl_${name}(const float *open, const float *high, const float *low, const float *close, float *output, int size, void *stream)`
    );
}

// ============================================================================
// Helpers
// ============================================================================

function toFloat32Array(arr) {
    if (arr instanceof Float32Array) return arr;
    return new Float32Array(arr);
}

/**
 * Multi-input indicators read every array at the same indices; passing
 * arrays of different lengths to the native side would read out of bounds.
 * Throws RangeError when lengths differ.
 */
function assertSameLength(arrays, what) {
    const n = arrays[0].length;
    for (let i = 1; i < arrays.length; ++i) {
        if (arrays[i].length !== n) {
            throw new RangeError(`${what}: input arrays must have equal length ` +
                `(got ${arrays.map(a => a.length).join(', ')})`);
        }
    }
}

// ============================================================================
// OHLCV columnar container
// ============================================================================

/**
 * Columnar OHLCV container backed by Float32Arrays.
 * Any plain object exposing the needed columns ({ high, low, close, ... })
 * is also accepted by the indicator functions; this class just provides
 * convenient constructors and guaranteed layout.
 */
class Ohlcv {
    constructor(size = 0) {
        if (!Number.isInteger(size) || size < 0) {
            throw new RangeError('Ohlcv size must be a non-negative integer');
        }
        this.size = size;
        this.open = new Float32Array(size);
        this.high = new Float32Array(size);
        this.low = new Float32Array(size);
        this.close = new Float32Array(size);
        this.volume = new Float32Array(size);
    }

    static fromColumns(open, high, low, close, volume = null) {
        const o = toFloat32Array(open);
        const h = toFloat32Array(high);
        const l = toFloat32Array(low);
        const c = toFloat32Array(close);
        if (h.length !== o.length || l.length !== o.length || c.length !== o.length) {
            throw new RangeError('Ohlcv columns must have equal length');
        }
        const inst = new Ohlcv(o.length);
        inst.open.set(o); inst.high.set(h); inst.low.set(l); inst.close.set(c);
        if (volume != null) {
            const v = toFloat32Array(volume);
            if (v.length !== o.length) throw new RangeError('volume column length mismatch');
            inst.volume.set(v);
        }
        return inst;
    }

    /** rows: array of [open, high, low, close(, volume)] or objects with those keys. */
    static fromRows(rows) {
        const inst = new Ohlcv(rows.length);
        for (let i = 0; i < rows.length; ++i) {
            const r = rows[i];
            if (Array.isArray(r) || ArrayBuffer.isView(r)) {
                inst.open[i] = r[0]; inst.high[i] = r[1];
                inst.low[i] = r[2]; inst.close[i] = r[3];
                inst.volume[i] = r.length > 4 ? r[4] : 0;
            } else {
                inst.open[i] = r.open; inst.high[i] = r.high;
                inst.low[i] = r.low; inst.close[i] = r.close;
                inst.volume[i] = r.volume ?? 0;
            }
        }
        return inst;
    }

    set(index, open, high, low, close, volume = 0) {
        if (index < 0 || index >= this.size) throw new RangeError('Ohlcv row index out of range');
        this.open[index] = open; this.high[index] = high;
        this.low[index] = low; this.close[index] = close;
        this.volume[index] = volume;
    }

    /** Column-major (SoA) packed Float32Array: [open..., high..., low..., close...(, volume...)]. */
    toColumnMajor(includeVolume = true) {
        const cols = includeVolume ? 5 : 4;
        const out = new Float32Array(cols * this.size);
        out.set(this.open, 0);
        out.set(this.high, this.size);
        out.set(this.low, 2 * this.size);
        out.set(this.close, 3 * this.size);
        if (includeVolume) out.set(this.volume, 4 * this.size);
        return out;
    }
}

// True for Ohlcv instances and plain column objects, false for array-likes.
function isOhlcvLike(value) {
    return value !== null && typeof value === 'object' &&
        !Array.isArray(value) && !ArrayBuffer.isView(value);
}

/**
 * Wrap an indicator so its leading column arguments can be replaced by a
 * single OHLCV container: atr(ohlcv, 14) === atr(high, low, close, 14).
 */
function ohlcvAware(fn, columns) {
    const wrapped = function (first, ...rest) {
        if (isOhlcvLike(first)) {
            const values = columns.map((c) => {
                const v = first[c];
                if (v == null) {
                    throw new TypeError(`OHLCV input is missing required column '${c}'`);
                }
                return v;
            });
            return fn(...values, ...rest);
        }
        return fn(first, ...rest);
    };
    Object.defineProperty(wrapped, 'name', { value: fn.name });
    return wrapped;
}

function call(fn, fnName, ...args) {
    const status = fn(...args, null);   // null = default CUDA stream
    if (status !== 0) throw new Error(`${fnName} failed with status ${status}`);
}

function singleOut(fn, fnName, input, ...extra) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(fn, fnName, a, out, a.length, ...extra);
    return out;
}

// ============================================================================
// Exported Wrapper Functions
// ============================================================================

// --- Moving Averages --------------------------------------------------------

function sma(input, period)  { return singleOut(ct_sma, 'ct_sma', input, period); }
function ema(input, period)  { return singleOut(ct_ema, 'ct_ema', input, period); }
function wma(input, period)  { return singleOut(ct_wma, 'ct_wma', input, period); }
function dema(input, period) { return singleOut(ct_dema, 'ct_dema', input, period); }
function tema(input, period) { return singleOut(ct_tema, 'ct_tema', input, period); }
function trima(input, period){ return singleOut(ct_trima, 'ct_trima', input, period); }

function t3(input, period, vFactor = 0.7) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_t3, 'ct_t3', a, out, a.length, period, vFactor);
    return out;
}

function kama(input, period = 10, fastPeriod = 2, slowPeriod = 30) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_kama, 'ct_kama', a, out, a.length, period, fastPeriod, slowPeriod);
    return out;
}

/** Generic MA with type enum: 0=SMA 1=EMA 2=WMA 3=DEMA 4=TEMA 5=TRIMA 6=KAMA 7=MAMA 8=T3 */
function ma(input, period, type = 0) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_ma, 'ct_ma', a, out, a.length, period, type);
    return out;
}

/** Variable-period MA */
function mavp(input, periods, minPeriod = 2, maxPeriod = 30, type = 0) {
    const a = toFloat32Array(input);
    const p = toFloat32Array(periods);
    const out = new Float32Array(a.length);
    call(ct_mavp, 'ct_mavp', a, p, out, a.length, minPeriod, maxPeriod, type);
    return out;
}

// --- Momentum ---------------------------------------------------------------

function momentum(input, period) { return singleOut(ct_momentum, 'ct_momentum', input, period); }
function change(input, period)   { return singleOut(ct_change, 'ct_change', input, period); }
function roc(input, period)      { return singleOut(ct_roc, 'ct_roc', input, period); }
function rocp(input, period)     { return singleOut(ct_rocp, 'ct_rocp', input, period); }
function rocr(input, period)     { return singleOut(ct_rocr, 'ct_rocr', input, period); }
function rocr100(input, period)  { return singleOut(ct_rocr100, 'ct_rocr100', input, period); }
function trix(input, period)     { return singleOut(ct_trix, 'ct_trix', input, period); }
function cmo(input, period)      { return singleOut(ct_cmo, 'ct_cmo', input, period); }
function rsi(input, period)      { return singleOut(ct_rsi, 'ct_rsi', input, period); }

function apo(input, fastPeriod = 12, slowPeriod = 26) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_apo, 'ct_apo', a, out, a.length, fastPeriod, slowPeriod);
    return out;
}

function ppo(input, fastPeriod = 12, slowPeriod = 26) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_ppo, 'ct_ppo', a, out, a.length, fastPeriod, slowPeriod);
    return out;
}

/**
 * APO with a selectable MA type (TA-Lib optInMAType; 0 = SMA is TA-Lib's
 * default, plain apo() keeps the historical EMA behavior).
 */
function apoMa(input, fastPeriod = 12, slowPeriod = 26, maType = 0) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_apo_ma, 'ct_apo_ma', a, out, a.length, fastPeriod, slowPeriod, maType);
    return out;
}

/** PPO with a selectable MA type (see apoMa). */
function ppoMa(input, fastPeriod = 12, slowPeriod = 26, maType = 0) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_ppo_ma, 'ct_ppo_ma', a, out, a.length, fastPeriod, slowPeriod, maType);
    return out;
}

function pvo(volume, fastPeriod = 12, slowPeriod = 26) {
    const a = toFloat32Array(volume);
    const out = new Float32Array(a.length);
    call(ct_pvo, 'ct_pvo', a, out, a.length, fastPeriod, slowPeriod);
    return out;
}

// --- MACD family ------------------------------------------------------------

function macdLine(input, fastPeriod = 12, slowPeriod = 26) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_macd_line, 'ct_macd_line', a, out, a.length, fastPeriod, slowPeriod);
    return out;
}

function macd(input, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9, maType = 0) {
    const a = toFloat32Array(input);
    const n = a.length;
    const m = new Float32Array(n), s = new Float32Array(n), h = new Float32Array(n);
    call(ct_macd, 'ct_macd', a, m, s, h, n, fastPeriod, slowPeriod, signalPeriod, maType);
    return { macd: m, signal: s, hist: h };
}

function macdfix(input, signalPeriod = 9) {
    const a = toFloat32Array(input);
    const n = a.length;
    const m = new Float32Array(n), s = new Float32Array(n), h = new Float32Array(n);
    call(ct_macdfix, 'ct_macdfix', a, m, s, h, n, signalPeriod);
    return { macd: m, signal: s, hist: h };
}

// --- MAMA -------------------------------------------------------------------

function mama(input, fastLimit = 0.5, slowLimit = 0.05) {
    const a = toFloat32Array(input);
    const n = a.length;
    const mamaOut = new Float32Array(n), famaOut = new Float32Array(n);
    call(ct_mama, 'ct_mama', a, mamaOut, famaOut, n, fastLimit, slowLimit);
    return { mama: mamaOut, fama: famaOut };
}

// --- Statistics --------------------------------------------------------------

function stddev(input, period)  { return singleOut(ct_stddev, 'ct_stddev', input, period); }
function avgdev(input, period)  { return singleOut(ct_avgdev, 'ct_avgdev', input, period); }
function variance(input, period){ return singleOut(ct_var, 'ct_var', input, period); }
function sum(input, period)     { return singleOut(ct_sum, 'ct_sum', input, period); }
function max(input, period)     { return singleOut(ct_max, 'ct_max', input, period); }
function min(input, period)     { return singleOut(ct_min, 'ct_min', input, period); }
function maxindex(input, period){ return singleOut(ct_maxindex, 'ct_maxindex', input, period); }
function minindex(input, period){ return singleOut(ct_minindex, 'ct_minindex', input, period); }
function midpoint(input, period){ return singleOut(ct_midpoint, 'ct_midpoint', input, period); }

function minmax(input, period) {
    const a = toFloat32Array(input);
    const mn = new Float32Array(a.length), mx = new Float32Array(a.length);
    call(ct_minmax, 'ct_minmax', a, mn, mx, a.length, period);
    return { min: mn, max: mx };
}

function minmaxindex(input, period) {
    const a = toFloat32Array(input);
    const mni = new Float32Array(a.length), mxi = new Float32Array(a.length);
    call(ct_minmaxindex, 'ct_minmaxindex', a, mni, mxi, a.length, period);
    return { minIdx: mni, maxIdx: mxi };
}

function beta(x, y, period) {
    const ax = toFloat32Array(x), ay = toFloat32Array(y);
    assertSameLength([ax, ay], 'beta');
    const out = new Float32Array(ax.length);
    call(ct_beta, 'ct_beta', ax, ay, out, ax.length, period);
    return out;
}

function correl(x, y, period) {
    const ax = toFloat32Array(x), ay = toFloat32Array(y);
    assertSameLength([ax, ay], 'correl');
    const out = new Float32Array(ax.length);
    call(ct_correl, 'ct_correl', ax, ay, out, ax.length, period);
    return out;
}

// --- Linear Regression ------------------------------------------------------

function linearreg(input, period)          { return singleOut(ct_linearreg, 'ct_linearreg', input, period); }
function linearregSlope(input, period)     { return singleOut(ct_linearreg_slope, 'ct_linearreg_slope', input, period); }
function linearregIntercept(input, period) { return singleOut(ct_linearreg_intercept, 'ct_linearreg_intercept', input, period); }
function linearregAngle(input, period)     { return singleOut(ct_linearreg_angle, 'ct_linearreg_angle', input, period); }
function tsf(input, period)                { return singleOut(ct_tsf, 'ct_tsf', input, period); }

// --- Volatility / Bands -----------------------------------------------------

function bbands(input, period = 20, upperMul = 2.0, lowerMul = 2.0) {
    const a = toFloat32Array(input);
    const n = a.length;
    const u = new Float32Array(n), m = new Float32Array(n), l = new Float32Array(n);
    call(ct_bbands, 'ct_bbands', a, u, m, l, n, period, upperMul, lowerMul);
    return { upper: u, middle: m, lower: l };
}

function accbands(high, low, close, period = 20) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'accbands');
    const n = h.length;
    const u = new Float32Array(n), m = new Float32Array(n), lo = new Float32Array(n);
    call(ct_accbands, 'ct_accbands', h, l, c, u, m, lo, n, period);
    return { upper: u, middle: m, lower: lo };
}

function atr(high, low, close, period, initial = 0.0) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'atr');
    const out = new Float32Array(h.length);
    call(ct_atr, 'ct_atr', h, l, c, out, h.length, period, initial);
    return out;
}

function natr(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'natr');
    const out = new Float32Array(h.length);
    call(ct_natr, 'ct_natr', h, l, c, out, h.length, period);
    return out;
}

function trange(high, low, close) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'trange');
    const out = new Float32Array(h.length);
    call(ct_trange, 'ct_trange', h, l, c, out, h.length);
    return out;
}

// --- Oscillators ------------------------------------------------------------

function stochastic(high, low, close, kPeriod = 14, dPeriod = 3) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'stochastic');
    const n = h.length;
    const kOut = new Float32Array(n), dOut = new Float32Array(n);
    call(ct_stochastic, 'ct_stochastic', h, l, c, kOut, dOut, n, kPeriod, dPeriod);
    return { k: kOut, d: dOut };
}

function stochf(high, low, close, kPeriod = 5, dPeriod = 3) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'stochf');
    const n = h.length;
    const kOut = new Float32Array(n), dOut = new Float32Array(n);
    call(ct_stochf, 'ct_stochf', h, l, c, kOut, dOut, n, kPeriod, dPeriod);
    return { k: kOut, d: dOut };
}

function stochrsi(input, rsiPeriod = 14, kPeriod = 5, dPeriod = 3) {
    const a = toFloat32Array(input);
    const n = a.length;
    const kOut = new Float32Array(n), dOut = new Float32Array(n);
    call(ct_stochrsi, 'ct_stochrsi', a, kOut, dOut, n, rsiPeriod, kPeriod, dPeriod);
    return { k: kOut, d: dOut };
}

function cci(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'cci');
    const out = new Float32Array(h.length);
    call(ct_cci, 'ct_cci', h, l, c, out, h.length, period);
    return out;
}

function willr(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'willr');
    const out = new Float32Array(h.length);
    call(ct_willr, 'ct_willr', h, l, c, out, h.length, period);
    return out;
}

function ultosc(high, low, close, shortPeriod = 7, mediumPeriod = 14, longPeriod = 28) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'ultosc');
    const out = new Float32Array(h.length);
    call(ct_ultosc, 'ct_ultosc', h, l, c, out, h.length, shortPeriod, mediumPeriod, longPeriod);
    return out;
}

// --- Directional Movement ---------------------------------------------------

function adx(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'adx');
    const out = new Float32Array(h.length);
    call(ct_adx, 'ct_adx', h, l, c, out, h.length, period);
    return out;
}

function adxr(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'adxr');
    const out = new Float32Array(h.length);
    call(ct_adxr, 'ct_adxr', h, l, c, out, h.length, period);
    return out;
}

function dx(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'dx');
    const out = new Float32Array(h.length);
    call(ct_dx, 'ct_dx', h, l, c, out, h.length, period);
    return out;
}

function plusDm(high, low, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'plusDm');
    const out = new Float32Array(h.length);
    call(ct_plus_dm, 'ct_plus_dm', h, l, out, h.length, period);
    return out;
}

function minusDm(high, low, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'minusDm');
    const out = new Float32Array(h.length);
    call(ct_minus_dm, 'ct_minus_dm', h, l, out, h.length, period);
    return out;
}

function plusDi(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'plusDi');
    const out = new Float32Array(h.length);
    call(ct_plus_di, 'ct_plus_di', h, l, c, out, h.length, period);
    return out;
}

function minusDi(high, low, close, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'minusDi');
    const out = new Float32Array(h.length);
    call(ct_minus_di, 'ct_minus_di', h, l, c, out, h.length, period);
    return out;
}

// --- Volume -----------------------------------------------------------------

function obv(close, volume) {
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([c, v], 'obv');
    const out = new Float32Array(c.length);
    call(ct_obv, 'ct_obv', c, v, out, c.length);
    return out;
}

function mfi(high, low, close, volume, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([h, l, c, v], 'mfi');
    const out = new Float32Array(h.length);
    call(ct_mfi, 'ct_mfi', h, l, c, v, out, h.length, period);
    return out;
}

function ad(high, low, close, volume) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([h, l, c, v], 'ad');
    const out = new Float32Array(h.length);
    call(ct_ad, 'ct_ad', h, l, c, v, out, h.length);
    return out;
}

function adosc(high, low, close, volume, shortPeriod = 3, longPeriod = 10) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([h, l, c, v], 'adosc');
    const out = new Float32Array(h.length);
    call(ct_adosc, 'ct_adosc', h, l, c, v, out, h.length, shortPeriod, longPeriod);
    return out;
}

function imi(open, close, period) {
    const o = toFloat32Array(open), c = toFloat32Array(close);
    assertSameLength([o, c], 'imi');
    const out = new Float32Array(o.length);
    call(ct_imi, 'ct_imi', o, c, out, o.length, period);
    return out;
}

function nvi(close, volume) {
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([c, v], 'nvi');
    const out = new Float32Array(c.length);
    call(ct_nvi, 'ct_nvi', c, v, out, c.length);
    return out;
}

function pvi(close, volume) {
    const c = toFloat32Array(close), v = toFloat32Array(volume);
    assertSameLength([c, v], 'pvi');
    const out = new Float32Array(c.length);
    call(ct_pvi, 'ct_pvi', c, v, out, c.length);
    return out;
}

// --- SAR --------------------------------------------------------------------

function sar(high, low, step = 0.02, maxAcceleration = 0.2) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'sar');
    const out = new Float32Array(h.length);
    call(ct_sar, 'ct_sar', h, l, out, h.length, step, maxAcceleration);
    return out;
}

function sarext(high, low, startValue = 0, offsetOnReverse = 0,
                accInitLong = 0.02, accLong = 0.02, accMaxLong = 0.2,
                accInitShort = 0.02, accShort = 0.02, accMaxShort = 0.2) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'sarext');
    const out = new Float32Array(h.length);
    call(ct_sarext, 'ct_sarext', h, l, out, h.length,
         startValue, offsetOnReverse,
         accInitLong, accLong, accMaxLong,
         accInitShort, accShort, accMaxShort);
    return out;
}

// --- STAD club pack (0.4.0) ---------------------------------------------------

const ct_asi = lib.func('int ct_asi(const float *open, const float *high, const float *low, const float *close, float *output, int size, void *stream)');
const ct_sar_el = lib.func('int ct_sar_el(const float *high, const float *low, const float *close, float *output, int size, float accFact, float maxAccel, void *stream)');
const ct_obtr = lib.func('int ct_obtr(const float *high, const float *low, const float *close, float *output, int size, void *stream)');
const ct_wobv = lib.func('int ct_wobv(const float *open, const float *high, const float *low, const float *close, const float *volume, float *output, int size, void *stream)');
const ct_hpi = lib.func('int ct_hpi(const float *high, const float *low, const float *volume, float *output, int size, float factor, void *stream)');
const ct_market_strength = lib.func('int ct_market_strength(const float *input, float *output, int size, int period, void *stream)');
const ct_trendscore = lib.func('int ct_trendscore(const float *input, float *output, int size, int lookback, void *stream)');
const ct_refdev = lib.func('int ct_refdev(const float *input, float *output, int size, int smaPeriod, int sumPeriod, void *stream)');
const ct_skew = lib.func('int ct_skew(const float *input, float *output, int size, int period, void *stream)');
const ct_kurtosis = lib.func('int ct_kurtosis(const float *input, float *output, int size, int period, void *stream)');
const ct_pivot_high = lib.func('int ct_pivot_high(const float *input, float *output, int size, int strength, void *stream)');
const ct_pivot_low = lib.func('int ct_pivot_low(const float *input, float *output, int size, int strength, void *stream)');
const ct_xaverage = lib.func('int ct_xaverage(const float *input, float *output, int size, int period, void *stream)');
const ct_rolling_mean = lib.func('int ct_rolling_mean(const float *input, float *output, int size, int period, void *stream)');
const ct_rolling_max = lib.func('int ct_rolling_max(const float *input, float *output, int size, int period, void *stream)');
const ct_rolling_min = lib.func('int ct_rolling_min(const float *input, float *output, int size, int period, void *stream)');
const ct_rolling_std = lib.func('int ct_rolling_std(const float *input, float *output, int size, int period, void *stream)');

function asi(open, high, low, close) {
    const o = toFloat32Array(open), h = toFloat32Array(high);
    const l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([o, h, l, c], 'asi');
    const out = new Float32Array(o.length);
    call(ct_asi, 'ct_asi', o, h, l, c, out, o.length);
    return out;
}

function sarEl(high, low, close, accFact = 0.02, maxAccel = 0.2) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'sarEl');
    const out = new Float32Array(h.length);
    call(ct_sar_el, 'ct_sar_el', h, l, c, out, h.length, accFact, maxAccel);
    return out;
}

function obtr(high, low, close) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'obtr');
    const out = new Float32Array(h.length);
    call(ct_obtr, 'ct_obtr', h, l, c, out, h.length);
    return out;
}

function wobv(open, high, low, close, volume) {
    const o = toFloat32Array(open), h = toFloat32Array(high);
    const l = toFloat32Array(low), c = toFloat32Array(close);
    const v = toFloat32Array(volume);
    assertSameLength([o, h, l, c, v], 'wobv');
    const out = new Float32Array(o.length);
    call(ct_wobv, 'ct_wobv', o, h, l, c, v, out, o.length);
    return out;
}

function hpi(high, low, volume, factor = 0.1) {
    const h = toFloat32Array(high), l = toFloat32Array(low), v = toFloat32Array(volume);
    assertSameLength([h, l, v], 'hpi');
    const out = new Float32Array(h.length);
    call(ct_hpi, 'ct_hpi', h, l, v, out, h.length, factor);
    return out;
}

function unaryClub(fn, fnName, input, param) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(fn, fnName, a, out, a.length, param);
    return out;
}

function marketStrength(input, period) { return unaryClub(ct_market_strength, 'ct_market_strength', input, period); }
function trendscore(input, lookback) { return unaryClub(ct_trendscore, 'ct_trendscore', input, lookback); }
function refdev(input, smaPeriod, sumPeriod) {
    const a = toFloat32Array(input);
    const out = new Float32Array(a.length);
    call(ct_refdev, 'ct_refdev', a, out, a.length, smaPeriod, sumPeriod);
    return out;
}
function skew(input, period) { return unaryClub(ct_skew, 'ct_skew', input, period); }
function kurtosis(input, period) { return unaryClub(ct_kurtosis, 'ct_kurtosis', input, period); }
function pivotHigh(input, strength) { return unaryClub(ct_pivot_high, 'ct_pivot_high', input, strength); }
function pivotLow(input, strength) { return unaryClub(ct_pivot_low, 'ct_pivot_low', input, strength); }
function xaverage(input, period) { return unaryClub(ct_xaverage, 'ct_xaverage', input, period); }
function rollingMean(input, period) { return unaryClub(ct_rolling_mean, 'ct_rolling_mean', input, period); }
function rollingMax(input, period) { return unaryClub(ct_rolling_max, 'ct_rolling_max', input, period); }
function rollingMin(input, period) { return unaryClub(ct_rolling_min, 'ct_rolling_min', input, period); }
function rollingStd(input, period) { return unaryClub(ct_rolling_std, 'ct_rolling_std', input, period); }

// --- MACDEXT ----------------------------------------------------------------

/** MA type enum shared by ma(), macd(), macdext(): 0=SMA 1=EMA 2=WMA 3=DEMA 4=TEMA 5=TRIMA 6=KAMA 7=MAMA 8=T3 */
const MA_TYPE = Object.freeze({
    SMA: 0, EMA: 1, WMA: 2, DEMA: 3, TEMA: 4, TRIMA: 5, KAMA: 6, MAMA: 7, T3: 8,
});

/** Storage/compute dtype enum for *_ex precision functions. */
const DTYPE = Object.freeze({ F16: 0, F32: 1, F64: 2 });

function macdext(input, fastPeriod = 12, fastType = 0, slowPeriod = 26, slowType = 0,
                 signalPeriod = 9, signalType = 0) {
    const a = toFloat32Array(input);
    const n = a.length;
    const m = new Float32Array(n), s = new Float32Array(n), h = new Float32Array(n);
    call(ct_macdext, 'ct_macdext', a, m, s, h, n,
         fastPeriod, fastType, slowPeriod, slowType, signalPeriod, signalType);
    return { macd: m, signal: s, hist: h };
}

// --- Vector Math ------------------------------------------------------------

function binaryOut(fn, fnName, input1, input2) {
    const a = toFloat32Array(input1), b = toFloat32Array(input2);
    assertSameLength([a, b], 'binaryOut');
    if (a.length !== b.length) {
        throw new Error(`${fnName}: input arrays must have the same length`);
    }
    const out = new Float32Array(a.length);
    call(fn, fnName, a, b, out, a.length);
    return out;
}

const vectorMath = {};
for (const name of UNARY_MATH_NAMES) {
    vectorMath[name] = (input) => singleOut(unaryMathFuncs[name], `ct_${name}`, input);
}
for (const name of BINARY_MATH_NAMES) {
    vectorMath[name] = (input1, input2) => binaryOut(binaryMathFuncs[name], `ct_${name}`, input1, input2);
}

// --- Precision (_ex) wrappers -----------------------------------------------

function dtypeOf(arr, fnName) {
    if (arr instanceof Float64Array) return DTYPE.F64;
    if (arr instanceof Float32Array) return DTYPE.F32;
    throw new Error(`${fnName}: pass Float32Array or Float64Array (float16 storage is not addressable from JS)`);
}

function toTyped(input) {
    if (input instanceof Float64Array || input instanceof Float32Array) return input;
    return new Float32Array(input);
}

function sameTyped(template, length) {
    return template instanceof Float64Array ? new Float64Array(length) : new Float32Array(length);
}

/**
 * Multi-input _ex functions: all arrays must share BOTH the length and the
 * TypedArray dtype — the native side reads every buffer with the single
 * storageType element size.
 */
function assertSameTyped(arrays, what) {
    assertSameLength(arrays, what);
    const ctor = arrays[0].constructor;
    for (let i = 1; i < arrays.length; ++i) {
        if (arrays[i].constructor !== ctor) {
            throw new TypeError(`${what}: all inputs must be the same TypedArray ` +
                `dtype (got ${arrays.map(a => a.constructor.name).join(', ')})`);
        }
    }
}

/** SMA with dtype selection: pass Float64Array for double-precision storage+compute. */
function smaEx(input, period) {
    const a = toTyped(input);
    const dt = dtypeOf(a, 'ct_sma_ex');
    const out = sameTyped(a, a.length);
    call(ct_sma_ex, 'ct_sma_ex', a, out, a.length, period, dt, dt);
    return out;
}

function emaEx(input, period) {
    const a = toTyped(input);
    const dt = dtypeOf(a, 'ct_ema_ex');
    const out = sameTyped(a, a.length);
    call(ct_ema_ex, 'ct_ema_ex', a, out, a.length, period, dt, dt);
    return out;
}

function rsiEx(input, period) {
    const a = toTyped(input);
    const dt = dtypeOf(a, 'ct_rsi_ex');
    const out = sameTyped(a, a.length);
    call(ct_rsi_ex, 'ct_rsi_ex', a, out, a.length, period, dt, dt);
    return out;
}

function macdEx(input, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9, maType = 1) {
    const a = toTyped(input);
    const dt = dtypeOf(a, 'ct_macd_ex');
    const n = a.length;
    const m = sameTyped(a, n), s = sameTyped(a, n), h = sameTyped(a, n);
    call(ct_macd_ex, 'ct_macd_ex', a, m, s, h, n, fastPeriod, slowPeriod, signalPeriod, maType, dt, dt);
    return { macd: m, signal: s, hist: h };
}

function atrEx(high, low, close, period, initial = 0.0) {
    const h = toTyped(high), l = toTyped(low), c = toTyped(close);
    assertSameTyped([h, l, c], 'atrEx');
    const dt = dtypeOf(h, 'ct_atr_ex');
    const out = sameTyped(h, h.length);
    call(ct_atr_ex, 'ct_atr_ex', h, l, c, out, h.length, period, initial, dt, dt);
    return out;
}

function bbandsEx(input, period = 20, upperMul = 2.0, lowerMul = 2.0) {
    const a = toTyped(input);
    const dt = dtypeOf(a, 'ct_bbands_ex');
    const n = a.length;
    const u = sameTyped(a, n), m = sameTyped(a, n), l = sameTyped(a, n);
    call(ct_bbands_ex, 'ct_bbands_ex', a, u, m, l, n, period, upperMul, lowerMul, dt, dt);
    return { upper: u, middle: m, lower: l };
}

// --- Device API (zero-copy GPU pipelines) ------------------------------------

function deviceCall(fn, fnName, ...args) {
    const status = fn(...args, null);
    if (status !== 0) throw new Error(`${fnName} failed with status ${status}`);
}

const device = {
    /** Allocate a GPU buffer of `count` float32 elements. Returns an opaque device pointer. */
    alloc(count) {
        const ptr = ct_device_alloc(count);
        if (!ptr) throw new Error('ct_device_alloc failed');
        return ptr;
    },
    /** Allocate a GPU buffer with explicit storage dtype (DTYPE.F16/F32/F64). */
    allocEx(count, storageType) {
        const ptr = ct_device_alloc_ex(count, storageType);
        if (!ptr) throw new Error('ct_device_alloc_ex failed');
        return ptr;
    },
    free(dPtr)   { ct_device_free(dPtr); },
    freeEx(dPtr) { ct_device_free_ex(dPtr); },
    /** Copy host Float32Array -> device buffer. */
    upload(dDst, hostSrc) {
        const a = toFloat32Array(hostSrc);
        deviceCall(ct_device_upload, 'ct_device_upload', dDst, a, a.length);
        return a.length;
    },
    /** Copy device buffer -> new host Float32Array of `count` elements. */
    download(dSrc, count) {
        const out = new Float32Array(count);
        deviceCall(ct_device_download, 'ct_device_download', out, dSrc, count);
        return out;
    },
    /** Copy typed host data (Float32Array/Float64Array) -> device buffer allocated via allocEx. */
    uploadEx(dDst, hostSrc) {
        const a = toTyped(hostSrc);
        deviceCall(ct_device_upload_ex, 'ct_device_upload_ex', dDst, a, a.length, dtypeOf(a, 'ct_device_upload_ex'));
        return a.length;
    },
    /** Copy device buffer -> new typed host array. storageType: DTYPE.F32 or DTYPE.F64. */
    downloadEx(dSrc, count, storageType) {
        const out = storageType === DTYPE.F64 ? new Float64Array(count) : new Float32Array(count);
        deviceCall(ct_device_download_ex, 'ct_device_download_ex', out, dSrc, count, storageType);
        return out;
    },
    sync() { deviceCall(ct_device_sync, 'ct_device_sync'); },

    smaEx(dIn, dOut, size, period, storageType, computeType = storageType) {
        deviceCall(ct_sma_ex_d, 'ct_sma_ex_d', dIn, dOut, size, period, storageType, computeType);
    },
    emaEx(dIn, dOut, size, period, storageType, computeType = storageType) {
        deviceCall(ct_ema_ex_d, 'ct_ema_ex_d', dIn, dOut, size, period, storageType, computeType);
    },
    rsiEx(dIn, dOut, size, period, storageType, computeType = storageType) {
        deviceCall(ct_rsi_ex_d, 'ct_rsi_ex_d', dIn, dOut, size, period, storageType, computeType);
    },
    atrEx(dHigh, dLow, dClose, dOut, size, period, initial, storageType, computeType = storageType) {
        deviceCall(ct_atr_ex_d, 'ct_atr_ex_d', dHigh, dLow, dClose, dOut, size, period, initial, storageType, computeType);
    },
    macdEx(dIn, dMacd, dSignal, dHist, size, fastPeriod, slowPeriod, signalPeriod, maType, storageType, computeType = storageType) {
        deviceCall(ct_macd_ex_d, 'ct_macd_ex_d', dIn, dMacd, dSignal, dHist, size, fastPeriod, slowPeriod, signalPeriod, maType, storageType, computeType);
    },
    bbandsEx(dIn, dUpper, dMiddle, dLower, size, period, nbDevUp, nbDevDn, storageType, computeType = storageType) {
        deviceCall(ct_bbands_ex_d, 'ct_bbands_ex_d', dIn, dUpper, dMiddle, dLower, size, period, nbDevUp, nbDevDn, storageType, computeType);
    },

    ma(dIn, dOut, size, period, type = 0) {
        deviceCall(ct_ma_device, 'ct_ma_device', dIn, dOut, size, period, type);
    },
    macdLine(dIn, dOut, size, fastPeriod = 12, slowPeriod = 26) {
        deviceCall(ct_macd_line_device, 'ct_macd_line_device', dIn, dOut, size, fastPeriod, slowPeriod);
    },
    apo(dIn, dOut, size, fastPeriod = 12, slowPeriod = 26) {
        deviceCall(ct_apo_device, 'ct_apo_device', dIn, dOut, size, fastPeriod, slowPeriod);
    },
    ppo(dIn, dOut, size, fastPeriod = 12, slowPeriod = 26) {
        deviceCall(ct_ppo_device, 'ct_ppo_device', dIn, dOut, size, fastPeriod, slowPeriod);
    },
    pvo(dVolume, dOut, size, fastPeriod = 12, slowPeriod = 26) {
        deviceCall(ct_pvo_device, 'ct_pvo_device', dVolume, dOut, size, fastPeriod, slowPeriod);
    },

    kama(dIn, dOut, size, period = 10, fastPeriod = 2, slowPeriod = 30) {
        deviceCall(ct_kama_d, 'ct_kama_d', dIn, dOut, size, period, fastPeriod, slowPeriod);
    },
    t3(dIn, dOut, size, period, vFactor = 0.7) {
        deviceCall(ct_t3_d, 'ct_t3_d', dIn, dOut, size, period, vFactor);
    },
    atr(dHigh, dLow, dClose, dOut, size, period, initial = 0.0) {
        deviceCall(ct_atr_d, 'ct_atr_d', dHigh, dLow, dClose, dOut, size, period, initial);
    },
    trange(dHigh, dLow, dClose, dOut, size) {
        deviceCall(ct_trange_d, 'ct_trange_d', dHigh, dLow, dClose, dOut, size);
    },
    macd(dIn, dMacd, dSignal, dHist, size, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9, maType = 1) {
        deviceCall(ct_macd_d, 'ct_macd_d', dIn, dMacd, dSignal, dHist, size, fastPeriod, slowPeriod, signalPeriod, maType);
    },
    macdext(dIn, dMacd, dSignal, dHist, size, fastPeriod = 12, fastType = 0, slowPeriod = 26, slowType = 0, signalPeriod = 9, signalType = 0) {
        deviceCall(ct_macdext_d, 'ct_macdext_d', dIn, dMacd, dSignal, dHist, size, fastPeriod, fastType, slowPeriod, slowType, signalPeriod, signalType);
    },
    stochastic(dHigh, dLow, dClose, dSlowK, dSlowD, size, kPeriod = 14, dPeriod = 3) {
        deviceCall(ct_stochastic_d, 'ct_stochastic_d', dHigh, dLow, dClose, dSlowK, dSlowD, size, kPeriod, dPeriod);
    },
    stochasticFast(dHigh, dLow, dClose, dFastK, dFastD, size, kPeriod = 5, dPeriod = 3, dMaType = 0) {
        deviceCall(ct_stochastic_fast_d, 'ct_stochastic_fast_d', dHigh, dLow, dClose, dFastK, dFastD, size, kPeriod, dPeriod, dMaType);
    },
    ultosc(dHigh, dLow, dClose, dOut, size, period1 = 7, period2 = 14, period3 = 28) {
        deviceCall(ct_ultosc_d, 'ct_ultosc_d', dHigh, dLow, dClose, dOut, size, period1, period2, period3);
    },
    bbands(dIn, dUpper, dMiddle, dLower, size, period = 20, nbDevUp = 2.0, nbDevDn = 2.0) {
        deviceCall(ct_bbands_d, 'ct_bbands_d', dIn, dUpper, dMiddle, dLower, size, period, nbDevUp, nbDevDn);
    },
    accbands(dHigh, dLow, dClose, dUpper, dMiddle, dLower, size, period = 20) {
        deviceCall(ct_accbands_d, 'ct_accbands_d', dHigh, dLow, dClose, dUpper, dMiddle, dLower, size, period);
    },
    ad(dHigh, dLow, dClose, dVolume, dOut, size) {
        deviceCall(ct_ad_d, 'ct_ad_d', dHigh, dLow, dClose, dVolume, dOut, size);
    },
    adosc(dHigh, dLow, dClose, dVolume, dOut, size, fastPeriod = 3, slowPeriod = 10) {
        deviceCall(ct_adosc_d, 'ct_adosc_d', dHigh, dLow, dClose, dVolume, dOut, size, fastPeriod, slowPeriod);
    },
    obv(dClose, dVolume, dOut, size) {
        deviceCall(ct_obv_d, 'ct_obv_d', dClose, dVolume, dOut, size);
    },
    mfi(dHigh, dLow, dClose, dVolume, dOut, size, period = 14) {
        deviceCall(ct_mfi_d, 'ct_mfi_d', dHigh, dLow, dClose, dVolume, dOut, size, period);
    },
    avgprice(dOpen, dHigh, dLow, dClose, dOut, size) {
        deviceCall(ct_avgprice_d, 'ct_avgprice_d', dOpen, dHigh, dLow, dClose, dOut, size);
    },
    medprice(dHigh, dLow, dOut, size) {
        deviceCall(ct_medprice_d, 'ct_medprice_d', dHigh, dLow, dOut, size);
    },
    typprice(dHigh, dLow, dClose, dOut, size) {
        deviceCall(ct_typprice_d, 'ct_typprice_d', dHigh, dLow, dClose, dOut, size);
    },
    wclprice(dHigh, dLow, dClose, dOut, size) {
        deviceCall(ct_wclprice_d, 'ct_wclprice_d', dHigh, dLow, dClose, dOut, size);
    },
    aroon(dHigh, dLow, dAroonDown, dAroonUp, size, period = 14) {
        deviceCall(ct_aroon_d, 'ct_aroon_d', dHigh, dLow, dAroonDown, dAroonUp, size, period);
    },
    aroonosc(dHigh, dLow, dOut, size, period = 14) {
        deviceCall(ct_aroonosc_d, 'ct_aroonosc_d', dHigh, dLow, dOut, size, period);
    },

    // --- Batched (multi-symbol) device API: series-major matrices on GPU ---
    smaBatch(dIn, dOut, seriesLen, numSeries, period, maxConcurrency = 0) {
        deviceCall(ct_sma_batch_d, 'ct_sma_batch_d', dIn, dOut, seriesLen, numSeries, period, maxConcurrency);
    },
    emaBatch(dIn, dOut, seriesLen, numSeries, period, maxConcurrency = 0) {
        deviceCall(ct_ema_batch_d, 'ct_ema_batch_d', dIn, dOut, seriesLen, numSeries, period, maxConcurrency);
    },
    rsiBatch(dIn, dOut, seriesLen, numSeries, period, maxConcurrency = 0) {
        deviceCall(ct_rsi_batch_d, 'ct_rsi_batch_d', dIn, dOut, seriesLen, numSeries, period, maxConcurrency);
    },
    atrBatch(dHighs, dLows, dCloses, dOut, seriesLen, numSeries, period, initial = 0.0, maxConcurrency = 0) {
        deviceCall(ct_atr_batch_d, 'ct_atr_batch_d', dHighs, dLows, dCloses, dOut, seriesLen, numSeries, period, initial, maxConcurrency);
    },
    macdBatch(dIn, dMacds, dSignals, dHists, seriesLen, numSeries, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9, maType = 1, maxConcurrency = 0) {
        deviceCall(ct_macd_batch_d, 'ct_macd_batch_d', dIn, dMacds, dSignals, dHists, seriesLen, numSeries, fastPeriod, slowPeriod, signalPeriod, maType, maxConcurrency);
    },
    bbandsBatch(dIn, dUppers, dMiddles, dLowers, seriesLen, numSeries, period, upperMul = 2.0, lowerMul = 2.0, maxConcurrency = 0) {
        deviceCall(ct_bbands_batch_d, 'ct_bbands_batch_d', dIn, dUppers, dMiddles, dLowers, seriesLen, numSeries, period, upperMul, lowerMul, maxConcurrency);
    },
    stochasticBatch(dHighs, dLows, dCloses, dKs, dDs, seriesLen, numSeries, kPeriod = 14, dPeriod = 3, maxConcurrency = 0) {
        deviceCall(ct_stochastic_batch_d, 'ct_stochastic_batch_d', dHighs, dLows, dCloses, dKs, dDs, seriesLen, numSeries, kPeriod, dPeriod, maxConcurrency);
    },
};

// Single-input + period indicators: device.sma(dIn, dOut, size, period), etc.
for (const name of DEVICE_PERIOD_FUNCS) {
    const jsName = name.replace(/_([a-z])/g, (_, ch) => ch.toUpperCase());
    device[jsName] = (dIn, dOut, size, period) =>
        deviceCall(devicePeriodFuncs[name], `ct_${name}_d`, dIn, dOut, size, period);
}
// HLC + period indicators: device.adx(dHigh, dLow, dClose, dOut, size, period), etc.
for (const name of DEVICE_HLC_PERIOD_FUNCS) {
    const jsName = name.replace(/_([a-z])/g, (_, ch) => ch.toUpperCase());
    device[jsName] = (dHigh, dLow, dClose, dOut, size, period) =>
        deviceCall(deviceHlcPeriodFuncs[name], `ct_${name}_d`, dHigh, dLow, dClose, dOut, size, period);
}
// Vector math on device pointers: device.sin(dIn, dOut, size), device.add(dA, dB, dOut, size)
for (const name of UNARY_MATH_NAMES) {
    device[name] = (dIn, dOut, size) =>
        deviceCall(deviceUnaryMathFuncs[name], `ct_${name}_d`, dIn, dOut, size);
}
for (const name of BINARY_MATH_NAMES) {
    device[name] = (dIn1, dIn2, dOut, size) =>
        deviceCall(deviceBinaryMathFuncs[name], `ct_${name}_d`, dIn1, dIn2, dOut, size);
}
// Legacy ct_*_device coverage: fill JS names that have no _d counterpart
for (const name of ['change', 'trix', 'maxindex', 'minindex']) {
    device[name] = (dIn, dOut, size, period) =>
        deviceCall(deviceLegacyPeriodFuncs[name], `ct_${name}_device`, dIn, dOut, size, period);
}
// Raw access to every legacy ct_*_device handle (advanced use)
device._legacy = { ...deviceLegacyPeriodFuncs, ma: ct_ma_device, kama: ct_kama_device, t3: ct_t3_device,
                   macd_line: ct_macd_line_device, apo: ct_apo_device, ppo: ct_ppo_device, pvo: ct_pvo_device };

// --- DeviceBufferPool control -------------------------------------------------

const bufferPool = {
    cleanup() { ct_device_buffer_pool_cleanup(); },
    allocationCount() { return Number(ct_device_buffer_pool_allocation_count()); },
    setMaxCacheBytes(bytes) { return Number(ct_device_buffer_pool_set_max_cache_bytes(bytes)); },
};

// --- Aroon ------------------------------------------------------------------

function aroon(high, low, upPeriod = 14, downPeriod = 14) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'aroon');
    const n = h.length;
    const up = new Float32Array(n), dn = new Float32Array(n), osc = new Float32Array(n);
    call(ct_aroon, 'ct_aroon', h, l, up, dn, osc, n, upPeriod, downPeriod);
    return { up, down: dn, osc };
}

function aroonosc(high, low, period = 14) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'aroonosc');
    const out = new Float32Array(h.length);
    call(ct_aroonosc, 'ct_aroonosc', h, l, out, h.length, period);
    return out;
}

// --- Price Transform --------------------------------------------------------

function avgprice(open, high, low, close) {
    const o = toFloat32Array(open), h = toFloat32Array(high);
    const l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([o, h, l, c], 'avgprice');
    const out = new Float32Array(h.length);
    call(ct_avgprice, 'ct_avgprice', o, h, l, c, out, h.length);
    return out;
}

function medprice(high, low) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'medprice');
    const out = new Float32Array(h.length);
    call(ct_medprice, 'ct_medprice', h, l, out, h.length);
    return out;
}

function typprice(high, low, close) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'typprice');
    const out = new Float32Array(h.length);
    call(ct_typprice, 'ct_typprice', h, l, c, out, h.length);
    return out;
}

function wclprice(high, low, close) {
    const h = toFloat32Array(high), l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([h, l, c], 'wclprice');
    const out = new Float32Array(h.length);
    call(ct_wclprice, 'ct_wclprice', h, l, c, out, h.length);
    return out;
}

function midprice(high, low, period) {
    const h = toFloat32Array(high), l = toFloat32Array(low);
    assertSameLength([h, l], 'midprice');
    const out = new Float32Array(h.length);
    call(ct_midprice, 'ct_midprice', h, l, out, h.length, period);
    return out;
}

function bop(open, high, low, close) {
    const o = toFloat32Array(open), h = toFloat32Array(high);
    const l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([o, h, l, c], 'bop');
    const out = new Float32Array(h.length);
    call(ct_bop, 'ct_bop', o, h, l, c, out, h.length);
    return out;
}

// --- Hilbert Transform ------------------------------------------------------

function htDcperiod(input)  { return singleOut(ct_ht_dcperiod, 'ct_ht_dcperiod', input); }
function htDcphase(input)   { return singleOut(ct_ht_dcphase, 'ct_ht_dcphase', input); }
function htTrendline(input) { return singleOut(ct_ht_trendline, 'ct_ht_trendline', input); }
function htTrendmode(input) { return singleOut(ct_ht_trendmode, 'ct_ht_trendmode', input); }

function htPhasor(input) {
    const a = toFloat32Array(input);
    const n = a.length;
    const ip = new Float32Array(n), q = new Float32Array(n);
    call(ct_ht_phasor, 'ct_ht_phasor', a, ip, q, n);
    return { inphase: ip, quadrature: q };
}

function htSine(input) {
    const a = toFloat32Array(input);
    const n = a.length;
    const s = new Float32Array(n), ls = new Float32Array(n);
    call(ct_ht_sine, 'ct_ht_sine', a, s, ls, n);
    return { sine: s, leadsine: ls };
}

// --- Candlestick Patterns ---------------------------------------------------

function cdlPattern(name, open, high, low, close) {
    const fn = cdlFuncs[name];
    if (!fn) throw new Error(`Unknown candlestick pattern: ${name}`);
    const o = toFloat32Array(open), h = toFloat32Array(high);
    const l = toFloat32Array(low), c = toFloat32Array(close);
    assertSameLength([o, h, l, c], 'cdlPattern');
    const out = new Float32Array(h.length);
    call(fn, `ct_cdl_${name}`, o, h, l, c, out, h.length);
    return out;
}

// Generate individual named exports for common patterns
function cdlDoji(o,h,l,c)             { return cdlPattern('doji',o,h,l,c); }
function cdlHammer(o,h,l,c)           { return cdlPattern('hammer',o,h,l,c); }
function cdlInvertedHammer(o,h,l,c)   { return cdlPattern('inverted_hammer',o,h,l,c); }
function cdlBullishEngulfing(o,h,l,c) { return cdlPattern('bullish_engulfing',o,h,l,c); }
function cdlBearishEngulfing(o,h,l,c) { return cdlPattern('bearish_engulfing',o,h,l,c); }
function cdlThreeWhiteSoldiers(o,h,l,c) { return cdlPattern('three_white_soldiers',o,h,l,c); }
function cdlThreeBlackCrows(o,h,l,c)  { return cdlPattern('three_black_crows',o,h,l,c); }
function cdlEngulfing(o,h,l,c)        { return cdlPattern('engulfing',o,h,l,c); }
function cdlMorningStar(o,h,l,c)      { return cdlPattern('morning_star',o,h,l,c); }
function cdlEveningStar(o,h,l,c)      { return cdlPattern('evening_star',o,h,l,c); }

// --- Batched (multi-symbol) API ----------------------------------------------

/**
 * Normalize batch input: accepts an array of per-series arrays, or a flat
 * Float32Array of numSeries*seriesLen elements together with {seriesLen}.
 * Returns { matrix: Float32Array, seriesLen, numSeries }.
 */
function toBatchMatrix(input, seriesLenHint) {
    if (Array.isArray(input) && input.length > 0 &&
        (Array.isArray(input[0]) || ArrayBuffer.isView(input[0]))) {
        const numSeries = input.length;
        const seriesLen = input[0].length;
        const matrix = new Float32Array(numSeries * seriesLen);
        for (let s = 0; s < numSeries; ++s) {
            if (input[s].length !== seriesLen) {
                throw new RangeError('all series in a batch must have the same length');
            }
            matrix.set(toFloat32Array(input[s]), s * seriesLen);
        }
        return { matrix, seriesLen, numSeries };
    }
    const flat = toFloat32Array(input);
    const seriesLen = seriesLenHint | 0;
    if (seriesLen <= 0 || flat.length % seriesLen !== 0) {
        throw new RangeError('flat batch input requires a valid seriesLen');
    }
    return { matrix: flat, seriesLen, numSeries: flat.length / seriesLen };
}

function splitBatchRows(matrix, seriesLen, numSeries) {
    const rows = new Array(numSeries);
    for (let s = 0; s < numSeries; ++s) {
        rows[s] = matrix.subarray(s * seriesLen, (s + 1) * seriesLen);
    }
    return rows;
}

function singleBatch(fn, fnName, inputs, opts, ...params) {
    const { matrix, seriesLen, numSeries } = toBatchMatrix(inputs, opts.seriesLen);
    const out = new Float32Array(matrix.length);
    call(fn, fnName, matrix, out, seriesLen, numSeries, ...params, opts.maxConcurrency ?? 0);
    return splitBatchRows(out, seriesLen, numSeries);
}

/** SMA per series; inputs = array of series or flat matrix with opts.seriesLen. */
function smaBatch(inputs, period, opts = {}) {
    return singleBatch(ct_sma_batch, 'ct_sma_batch', inputs, opts, period);
}

function emaBatch(inputs, period, opts = {}) {
    return singleBatch(ct_ema_batch, 'ct_ema_batch', inputs, opts, period);
}

function rsiBatch(inputs, period, opts = {}) {
    return singleBatch(ct_rsi_batch, 'ct_rsi_batch', inputs, opts, period);
}

// Both dimensions must match: a 2x10 and a 4x5 matrix have the same element
// count but are different shapes.
function assertSameBatchShape(base, others, what) {
    for (const m of others) {
        if (m.seriesLen !== base.seriesLen || m.numSeries !== base.numSeries) {
            throw new RangeError(`${what} batches must have the same shape (seriesLen x numSeries)`);
        }
    }
}

function atrBatch(highs, lows, closes, period, opts = {}) {
    const h = toBatchMatrix(highs, opts.seriesLen);
    const l = toBatchMatrix(lows, opts.seriesLen);
    const c = toBatchMatrix(closes, opts.seriesLen);
    assertSameBatchShape(h, [l, c], 'high/low/close');
    const out = new Float32Array(h.matrix.length);
    call(ct_atr_batch, 'ct_atr_batch', h.matrix, l.matrix, c.matrix, out,
        h.seriesLen, h.numSeries, period, opts.initial ?? 0.0, opts.maxConcurrency ?? 0);
    return splitBatchRows(out, h.seriesLen, h.numSeries);
}

function macdBatch(inputs, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9, opts = {}) {
    const { matrix, seriesLen, numSeries } = toBatchMatrix(inputs, opts.seriesLen);
    const m = new Float32Array(matrix.length);
    const s = new Float32Array(matrix.length);
    const h = new Float32Array(matrix.length);
    call(ct_macd_batch, 'ct_macd_batch', matrix, m, s, h, seriesLen, numSeries,
        fastPeriod, slowPeriod, signalPeriod, opts.maType ?? 1, opts.maxConcurrency ?? 0);
    return {
        macd: splitBatchRows(m, seriesLen, numSeries),
        signal: splitBatchRows(s, seriesLen, numSeries),
        hist: splitBatchRows(h, seriesLen, numSeries),
    };
}

function bbandsBatch(inputs, period, upperMul = 2.0, lowerMul = 2.0, opts = {}) {
    const { matrix, seriesLen, numSeries } = toBatchMatrix(inputs, opts.seriesLen);
    const up = new Float32Array(matrix.length);
    const mid = new Float32Array(matrix.length);
    const lo = new Float32Array(matrix.length);
    call(ct_bbands_batch, 'ct_bbands_batch', matrix, up, mid, lo, seriesLen,
        numSeries, period, upperMul, lowerMul, opts.maxConcurrency ?? 0);
    return {
        upper: splitBatchRows(up, seriesLen, numSeries),
        middle: splitBatchRows(mid, seriesLen, numSeries),
        lower: splitBatchRows(lo, seriesLen, numSeries),
    };
}

function stochasticBatch(highs, lows, closes, kPeriod = 14, dPeriod = 3, opts = {}) {
    const h = toBatchMatrix(highs, opts.seriesLen);
    const l = toBatchMatrix(lows, opts.seriesLen);
    const c = toBatchMatrix(closes, opts.seriesLen);
    assertSameBatchShape(h, [l, c], 'high/low/close');
    const k = new Float32Array(h.matrix.length);
    const d = new Float32Array(h.matrix.length);
    call(ct_stochastic_batch, 'ct_stochastic_batch', h.matrix, l.matrix, c.matrix,
        k, d, h.seriesLen, h.numSeries, kPeriod, dPeriod, opts.maxConcurrency ?? 0);
    return {
        k: splitBatchRows(k, h.seriesLen, h.numSeries),
        d: splitBatchRows(d, h.seriesLen, h.numSeries),
    };
}

/** Native library version string. */
function version() {
    return ct_version();
}

// ============================================================================
// OHLCV-aware wrapping: these functions accept an Ohlcv (or plain object with
// the listed columns) as their first argument in place of separate arrays.
// ============================================================================

const OHLCV_COLUMNS = {
    accbands: ['high', 'low', 'close'],
    atr: ['high', 'low', 'close'],
    natr: ['high', 'low', 'close'],
    trange: ['high', 'low', 'close'],
    stochastic: ['high', 'low', 'close'],
    stochf: ['high', 'low', 'close'],
    cci: ['high', 'low', 'close'],
    willr: ['high', 'low', 'close'],
    ultosc: ['high', 'low', 'close'],
    adx: ['high', 'low', 'close'],
    adxr: ['high', 'low', 'close'],
    dx: ['high', 'low', 'close'],
    plusDm: ['high', 'low'],
    minusDm: ['high', 'low'],
    plusDi: ['high', 'low', 'close'],
    minusDi: ['high', 'low', 'close'],
    obv: ['close', 'volume'],
    mfi: ['high', 'low', 'close', 'volume'],
    ad: ['high', 'low', 'close', 'volume'],
    adosc: ['high', 'low', 'close', 'volume'],
    imi: ['open', 'close'],
    nvi: ['close', 'volume'],
    pvi: ['close', 'volume'],
    sar: ['high', 'low'],
    sarext: ['high', 'low'],
    aroon: ['high', 'low'],
    aroonosc: ['high', 'low'],
    avgprice: ['open', 'high', 'low', 'close'],
    medprice: ['high', 'low'],
    typprice: ['high', 'low', 'close'],
    wclprice: ['high', 'low', 'close'],
    midprice: ['high', 'low'],
    bop: ['open', 'high', 'low', 'close'],
    // STAD club pack
    asi: ['open', 'high', 'low', 'close'],
    sarEl: ['high', 'low', 'close'],
    obtr: ['high', 'low', 'close'],
    wobv: ['open', 'high', 'low', 'close', 'volume'],
    hpi: ['high', 'low', 'volume'],
};

// ============================================================================
// Module Exports
// ============================================================================

module.exports = {
    // Moving Averages
    sma, ema, wma, dema, tema, trima, t3, kama, ma, mavp,
    // Momentum
    momentum, change, roc, rocp, rocr, rocr100, trix, cmo, rsi, apo, ppo, apoMa, ppoMa, pvo,
    // MACD
    macdLine, macd, macdfix, macdext, mama,
    // Statistics
    stddev, avgdev, variance, sum, max, min, maxindex, minindex,
    midpoint, minmax, minmaxindex, beta, correl,
    // Linear Regression
    linearreg, linearregSlope, linearregIntercept, linearregAngle, tsf,
    // Volatility / Bands
    bbands, accbands, atr, natr, trange,
    // Oscillators
    stochastic, stochf, stochrsi, cci, willr, ultosc,
    // Directional Movement
    adx, adxr, dx, plusDm, minusDm, plusDi, minusDi,
    // Volume
    obv, mfi, ad, adosc, imi, nvi, pvi,
    // SAR
    sar, sarext,
    // STAD club pack (0.4.0)
    asi, sarEl, obtr, wobv, hpi, marketStrength, trendscore, refdev,
    skew, kurtosis, pivotHigh, pivotLow, xaverage,
    rollingMean, rollingMax, rollingMin, rollingStd,
    // Aroon
    aroon, aroonosc,
    // Price Transform
    avgprice, medprice, typprice, wclprice, midprice, bop,
    // Hilbert Transform
    htDcperiod, htDcphase, htPhasor, htSine, htTrendline, htTrendmode,
    // Vector math (element-wise): acos, asin, atan, ceil, cos, cosh, exp,
    // floor, ln, log10, sin, sinh, sqrt, tan, tanh, add, sub, mult, div
    ...vectorMath,
    // Precision (_ex) variants: Float64Array in -> Float64Array out
    smaEx, emaEx, rsiEx, macdEx, atrEx, bbandsEx,
    // Device API (raw GPU pointers, zero-copy pipelines)
    device,
    bufferPool,
    // Candlestick patterns (generic + named)
    cdlPattern,
    cdlDoji, cdlHammer, cdlInvertedHammer, cdlBullishEngulfing, cdlBearishEngulfing,
    cdlThreeWhiteSoldiers, cdlThreeBlackCrows, cdlEngulfing, cdlMorningStar, cdlEveningStar,
    // OHLCV container + structured input support
    Ohlcv,
    // Batched (multi-symbol) host API
    smaBatch, emaBatch, rsiBatch, atrBatch, macdBatch, bbandsBatch, stochasticBatch,
    // Metadata / enums
    version,
    CDL_NAMES,
    MA_TYPE,
    DTYPE,
    UNARY_MATH_NAMES,
    BINARY_MATH_NAMES,
    _libPath: libPath,
};

// Rebind OHLC(V) family exports so they also accept a single container
// (Ohlcv instance or plain { open, high, low, close, volume } object).
for (const [name, columns] of Object.entries(OHLCV_COLUMNS)) {
    module.exports[name] = ohlcvAware(module.exports[name], columns);
}
module.exports.cdlPattern = function (name, open, high, low, close) {
    if (isOhlcvLike(open)) {
        const s = open;
        return cdlPattern(name, s.open, s.high, s.low, s.close);
    }
    return cdlPattern(name, open, high, low, close);
};
for (const name of ['cdlDoji', 'cdlHammer', 'cdlInvertedHammer', 'cdlBullishEngulfing',
    'cdlBearishEngulfing', 'cdlThreeWhiteSoldiers', 'cdlThreeBlackCrows',
    'cdlEngulfing', 'cdlMorningStar', 'cdlEveningStar']) {
    module.exports[name] = ohlcvAware(module.exports[name], ['open', 'high', 'low', 'close']);
}
// Batch wrappers accept an array of Ohlcv containers as well:
// atrBatch(ohlcvArray, period, opts) / stochasticBatch(ohlcvArray, k, d, opts).
module.exports.atrBatch = function (highs, lows, closes, period, opts = {}) {
    if (Array.isArray(highs) && highs.length > 0 && isOhlcvLike(highs[0])) {
        const series = highs;
        const p = typeof lows === 'number' ? lows : 14;
        const o = (closes !== null && typeof closes === 'object') ? closes : {};
        return atrBatch(series.map(s => s.high), series.map(s => s.low),
            series.map(s => s.close), p, o);
    }
    return atrBatch(highs, lows, closes, period, opts);
};
module.exports.stochasticBatch = function (highs, lows, closes, kPeriod = 14, dPeriod = 3, opts = {}) {
    if (Array.isArray(highs) && highs.length > 0 && isOhlcvLike(highs[0])) {
        const series = highs;
        const k = typeof lows === 'number' ? lows : 14;
        const d = typeof closes === 'number' ? closes : 3;
        const o = (kPeriod !== null && typeof kPeriod === 'object') ? kPeriod : {};
        return stochasticBatch(series.map(s => s.high), series.map(s => s.low),
            series.map(s => s.close), k, d, o);
    }
    return stochasticBatch(highs, lows, closes, kPeriod, dPeriod, opts);
};

// Generated batch families (scripts/generate_batch_api.py): remaining
// ct_*_batch[_d] indicator families with the same matrix layout and
// scheduling semantics as the hand-written wrappers above.
function ohlcvBatchAware(fn, columns) {
    const wrapped = function (first, ...rest) {
        if (Array.isArray(first) && first.length > 0 && isOhlcvLike(first[0])) {
            const mats = columns.map(c => first.map(s => s[c]));
            return fn(...mats, ...rest);
        }
        return fn(first, ...rest);
    };
    Object.defineProperty(wrapped, 'name', { value: fn.name });
    return wrapped;
}

const generatedBatch = require('./batch_generated')({
    lib, call, deviceCall, toBatchMatrix, splitBatchRows,
});
for (const [name, fn] of Object.entries(generatedBatch.host)) {
    const cols = generatedBatch.ohlcvColumns[name];
    module.exports[name] = cols ? ohlcvBatchAware(fn, cols) : fn;
}
// Merge generated device wrappers (batch + single-series ct_*_d) without
// overriding hand-written ones (e.g. device.kama, device.atr).
for (const [name, fn] of Object.entries(generatedBatch.device)) {
    if (!(name in device)) device[name] = fn;
}
