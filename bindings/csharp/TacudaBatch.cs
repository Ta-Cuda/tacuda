// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;

namespace Tacuda.Bindings;

/// <summary>
/// Batched (multi-symbol / portfolio) helpers. Each method accepts jagged
/// arrays (one row per symbol, equal lengths), flattens them into the
/// series-major matrices required by the native <c>ct_*_batch</c> entry
/// points and splits results back into per-symbol rows.
/// </summary>
public static partial class TacudaBatch
{
    public static float[][] Sma(float[][] inputs, int period, int maxConcurrency = 0)
        => RunSingle(inputs, (matrix, output, len, count) =>
            NativeMethods.ct_sma_batch(matrix, output, len, count, period, maxConcurrency),
            nameof(NativeMethods.ct_sma_batch));

    public static float[][] Ema(float[][] inputs, int period, int maxConcurrency = 0)
        => RunSingle(inputs, (matrix, output, len, count) =>
            NativeMethods.ct_ema_batch(matrix, output, len, count, period, maxConcurrency),
            nameof(NativeMethods.ct_ema_batch));

    public static float[][] Rsi(float[][] inputs, int period, int maxConcurrency = 0)
        => RunSingle(inputs, (matrix, output, len, count) =>
            NativeMethods.ct_rsi_batch(matrix, output, len, count, period, maxConcurrency),
            nameof(NativeMethods.ct_rsi_batch));

    public static float[][] Atr(OhlcvSeries[] series, int period = 14, float initial = 0f, int maxConcurrency = 0)
    {
        var (len, count) = Shape(series);
        var highs = Flatten(series, s => s.High, len);
        var lows = Flatten(series, s => s.Low, len);
        var closes = Flatten(series, s => s.Close, len);
        var output = new float[len * count];
        NativeMethods.ThrowOnError(
            NativeMethods.ct_atr_batch(highs, lows, closes, output, len, count, period, initial, maxConcurrency),
            nameof(NativeMethods.ct_atr_batch));
        return Split(output, len, count);
    }

    public static (float[][] Macd, float[][] Signal, float[][] Hist) Macd(
        float[][] inputs, int fastPeriod = 12, int slowPeriod = 26,
        int signalPeriod = 9, int maType = 1, int maxConcurrency = 0)
    {
        var (len, count, matrix) = FlattenRows(inputs);
        var macd = new float[matrix.Length];
        var signal = new float[matrix.Length];
        var hist = new float[matrix.Length];
        NativeMethods.ThrowOnError(
            NativeMethods.ct_macd_batch(matrix, macd, signal, hist, len, count,
                fastPeriod, slowPeriod, signalPeriod, maType, maxConcurrency),
            nameof(NativeMethods.ct_macd_batch));
        return (Split(macd, len, count), Split(signal, len, count), Split(hist, len, count));
    }

    public static (float[][] Upper, float[][] Middle, float[][] Lower) Bbands(
        float[][] inputs, int period, float upperMul = 2f, float lowerMul = 2f, int maxConcurrency = 0)
    {
        var (len, count, matrix) = FlattenRows(inputs);
        var upper = new float[matrix.Length];
        var middle = new float[matrix.Length];
        var lower = new float[matrix.Length];
        NativeMethods.ThrowOnError(
            NativeMethods.ct_bbands_batch(matrix, upper, middle, lower, len, count,
                period, upperMul, lowerMul, maxConcurrency),
            nameof(NativeMethods.ct_bbands_batch));
        return (Split(upper, len, count), Split(middle, len, count), Split(lower, len, count));
    }

    public static (float[][] K, float[][] D) Stochastic(
        OhlcvSeries[] series, int kPeriod = 14, int dPeriod = 3, int maxConcurrency = 0)
    {
        var (len, count) = Shape(series);
        var highs = Flatten(series, s => s.High, len);
        var lows = Flatten(series, s => s.Low, len);
        var closes = Flatten(series, s => s.Close, len);
        var k = new float[len * count];
        var d = new float[len * count];
        NativeMethods.ThrowOnError(
            NativeMethods.ct_stochastic_batch(highs, lows, closes, k, d, len, count,
                kPeriod, dPeriod, maxConcurrency),
            nameof(NativeMethods.ct_stochastic_batch));
        return (Split(k, len, count), Split(d, len, count));
    }

    private static float[][] RunSingle(float[][] inputs, Func<float[], float[], int, int, int> invoke, string name)
    {
        var (len, count, matrix) = FlattenRows(inputs);
        var output = new float[matrix.Length];
        NativeMethods.ThrowOnError(invoke(matrix, output, len, count), name);
        return Split(output, len, count);
    }

    private static (int Len, int Count, float[] Matrix) FlattenRows(float[][] rows)
    {
        if (rows is null || rows.Length == 0)
        {
            throw new ArgumentException("batch input must contain at least one series", nameof(rows));
        }
        int len = rows[0].Length;
        var matrix = new float[len * rows.Length];
        for (int s = 0; s < rows.Length; ++s)
        {
            if (rows[s].Length != len)
            {
                throw new ArgumentException("all series in a batch must have the same length", nameof(rows));
            }
            Array.Copy(rows[s], 0, matrix, s * len, len);
        }
        return (len, rows.Length, matrix);
    }

    private static (int Len, int Count) Shape(OhlcvSeries[] series)
    {
        if (series is null || series.Length == 0)
        {
            throw new ArgumentException("batch input must contain at least one series", nameof(series));
        }
        int len = series[0].Length;
        foreach (var s in series)
        {
            if (s.Length != len)
            {
                throw new ArgumentException("all series in a batch must have the same length", nameof(series));
            }
        }
        return (len, series.Length);
    }

    private static float[] Flatten(OhlcvSeries[] series, Func<OhlcvSeries, float[]> column, int len)
    {
        var matrix = new float[len * series.Length];
        for (int s = 0; s < series.Length; ++s)
        {
            Array.Copy(column(series[s]), 0, matrix, s * len, len);
        }
        return matrix;
    }

    private static float[][] Split(float[] matrix, int len, int count)
    {
        var rows = new float[count][];
        for (int s = 0; s < count; ++s)
        {
            rows[s] = new float[len];
            Array.Copy(matrix, s * len, rows[s], 0, len);
        }
        return rows;
    }
}
