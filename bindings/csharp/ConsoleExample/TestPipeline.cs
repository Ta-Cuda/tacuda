// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;
using System.Linq;
using Tacuda.Bindings;

namespace Tacuda.Tests;

/// <summary>
/// Pipeline test: one data source, multiple indicators.
/// Similar to C++ test_context_pipeline.cpp
/// </summary>
public static class TestPipeline
{
    private const float Tolerance = 1e-4f;
    
    public static int RunPipelineTest()
    {
        Console.WriteLine("=" + new string('=', 69));
        Console.WriteLine("TACUDA C# Pipeline Test (Multi-Indicator)");
        Console.WriteLine("=" + new string('=', 69));
        
        const int N = 4096;
        
        // Generate synthetic OHLCV data
        var (open, high, low, close, volume) = GenerateOHLCV(N);
        
        Console.WriteLine($"Data size: {N} samples");
        Console.WriteLine();
        
        int passed = 0;
        int failed = 0;
        
        // ====================================================================
        // Calculate multiple indicators on the SAME data source
        // ====================================================================
        
        // RSI
        var rsi = new float[N];
        int status = NativeMethods.ct_rsi(close, rsi, N, period: 14);
        TestResult("RSI(14)", status == 0 && ValidateRSI(rsi), ref passed, ref failed);
        
        // ATR
        var atr = new float[N];
        status = NativeMethods.ct_atr(high, low, close, atr, N, period: 14, initial: 0f);
        TestResult("ATR(14)", status == 0 && ValidateNonNegative(atr, "ATR"), ref passed, ref failed);
        
        // ADX
        var adx = new float[N];
        status = NativeMethods.ct_adx(high, low, close, adx, N, period: 14);
        TestResult("ADX(14)", status == 0 && ValidateInRange(adx, 0, 100, "ADX"), ref passed, ref failed);
        
        // OBV
        var obv = new float[N];
        status = NativeMethods.ct_obv(close, volume, obv, N);
        TestResult("OBV", status == 0, ref passed, ref failed);
        
        // MFI
        var mfi = new float[N];
        status = NativeMethods.ct_mfi(high, low, close, volume, mfi, N, period: 14);
        TestResult("MFI(14)", status == 0 && ValidateInRange(mfi, 0, 100, "MFI"), ref passed, ref failed);
        
        // EMA (fast)
        var emaFast = new float[N];
        status = NativeMethods.ct_ema(close, emaFast, N, period: 12);
        TestResult("EMA(12)", status == 0, ref passed, ref failed);
        
        // EMA (slow)
        var emaSlow = new float[N];
        status = NativeMethods.ct_ema(close, emaSlow, N, period: 26);
        TestResult("EMA(26)", status == 0, ref passed, ref failed);
        
        // Trend = EMA(12) - EMA(26)
        var trend = new float[N];
        for (int i = 0; i < N; i++)
        {
            if (!float.IsNaN(emaFast[i]) && !float.IsNaN(emaSlow[i]))
                trend[i] = emaFast[i] - emaSlow[i];
            else
                trend[i] = float.NaN;
        }
        TestResult("Trend (EMA12-EMA26)", true, ref passed, ref failed);
        
        // MACD
        var macd = new float[N];
        var signal = new float[N];
        var hist = new float[N];
        status = NativeMethods.ct_macd(close, macd, signal, hist, N, 
                                        fastPeriod: 12, slowPeriod: 26, signalPeriod: 9, type: 0);
        TestResult("MACD(12,26,9)", status == 0, ref passed, ref failed);
        
        // Stochastic
        var stochK = new float[N];
        var stochD = new float[N];
        status = NativeMethods.ct_stochastic(high, low, close, stochK, stochD, N, 
                                              kPeriod: 14, dPeriod: 3);
        TestResult("Stochastic(14,3)", status == 0 && ValidateInRange(stochK, 0, 100, "StochK") 
                                                   && ValidateInRange(stochD, 0, 100, "StochD"), 
                   ref passed, ref failed);
        
        // Bollinger Bands
        var bbUpper = new float[N];
        var bbMiddle = new float[N];
        var bbLower = new float[N];
        status = NativeMethods.ct_bbands(close, bbUpper, bbMiddle, bbLower, N, 
                                          period: 20, upperMul: 2.0f, lowerMul: 2.0f);
        TestResult("BBANDS(20,2,2)", status == 0 && ValidateBBands(bbUpper, bbMiddle, bbLower), 
                   ref passed, ref failed);
        
        // SMA
        var sma = new float[N];
        status = NativeMethods.ct_sma(close, sma, N, period: 20);
        TestResult("SMA(20)", status == 0, ref passed, ref failed);
        
        // WMA
        var wma = new float[N];
        status = NativeMethods.ct_wma(close, wma, N, period: 10);
        TestResult("WMA(10)", status == 0, ref passed, ref failed);
        
        // StdDev
        var stddev = new float[N];
        status = NativeMethods.ct_stddev(close, stddev, N, period: 20);
        TestResult("StdDev(20)", status == 0 && ValidateNonNegative(stddev, "StdDev"), ref passed, ref failed);
        
        // Sum
        var sum = new float[N];
        status = NativeMethods.ct_sum(close, sum, N, period: 10);
        TestResult("Sum(10)", status == 0, ref passed, ref failed);
        
        // ====================================================================
        // OHLCV container: structured inputs must match column-array calls
        // ====================================================================
        var candles = new OhlcvSeries(open, high, low, close, volume);
        TestResult("OhlcvSeries.Atr == ct_atr", SeriesMatch(candles.Atr(14), atr), ref passed, ref failed);
        TestResult("OhlcvSeries.Adx == ct_adx", SeriesMatch(candles.Adx(14), adx), ref passed, ref failed);
        TestResult("OhlcvSeries.Mfi == ct_mfi", SeriesMatch(candles.Mfi(14), mfi), ref passed, ref failed);
        TestResult("OhlcvSeries.Obv == ct_obv", SeriesMatch(candles.Obv(), obv), ref passed, ref failed);
        var (extK, extD) = candles.Stochastic(14, 3);
        TestResult("OhlcvSeries.Stochastic == ct_stochastic",
                   SeriesMatch(extK, stochK) && SeriesMatch(extD, stochD), ref passed, ref failed);

        // ====================================================================
        // Batched (multi-symbol) API vs per-series calls
        // ====================================================================
        const int NumSeries = 4;
        const int SeriesLen = 1024;
        var batchInputs = new float[NumSeries][];
        for (int s = 0; s < NumSeries; s++)
        {
            var (_, _, _, batchClose, _) = GenerateOHLCV(SeriesLen);
            for (int i = 0; i < SeriesLen; i++)
            {
                batchClose[i] += s * 3f;
            }
            batchInputs[s] = batchClose;
        }

        bool batchOk = true;
        var rsiBatch = TacudaBatch.Rsi(batchInputs, 14, maxConcurrency: 3);
        for (int s = 0; s < NumSeries && batchOk; s++)
        {
            var single = new float[SeriesLen];
            NativeMethods.ct_rsi(batchInputs[s], single, SeriesLen, 14);
            batchOk = SeriesMatch(rsiBatch[s], single);
        }
        TestResult($"TacudaBatch.Rsi ({NumSeries} series x {SeriesLen})", batchOk, ref passed, ref failed);

        var (macdBatch, signalBatch, _) = TacudaBatch.Macd(batchInputs);
        var macdSingle = new float[SeriesLen];
        var signalSingle = new float[SeriesLen];
        var histSingle = new float[SeriesLen];
        NativeMethods.ct_macd(batchInputs[0], macdSingle, signalSingle, histSingle, SeriesLen, 12, 26, 9, 1);
        TestResult("TacudaBatch.Macd row 0 == ct_macd",
                   SeriesMatch(macdBatch[0], macdSingle) && SeriesMatch(signalBatch[0], signalSingle),
                   ref passed, ref failed);

        // ====================================================================
        // Generated batch families (0.3.0)
        // ====================================================================
        var batchHighs = new float[NumSeries][];
        var batchLows = new float[NumSeries][];
        var batchVolumes = new float[NumSeries][];
        for (int s = 0; s < NumSeries; s++)
        {
            batchHighs[s] = new float[SeriesLen];
            batchLows[s] = new float[SeriesLen];
            batchVolumes[s] = new float[SeriesLen];
            for (int i = 0; i < SeriesLen; i++)
            {
                batchHighs[s][i] = batchInputs[s][i] + 1f;
                batchLows[s][i] = batchInputs[s][i] - 1f;
                batchVolumes[s][i] = 1000f + 500f * MathF.Abs(MathF.Sin(i));
            }
        }

        bool genOk = true;
        var wmaBatch = TacudaBatch.Wma(batchInputs, 10, maxConcurrency: 2);
        for (int s = 0; s < NumSeries && genOk; s++)
        {
            var single = new float[SeriesLen];
            NativeMethods.ct_wma(batchInputs[s], single, SeriesLen, 10);
            genOk = SeriesMatch(wmaBatch[s], single);
        }
        TestResult("TacudaBatch.Wma matches ct_wma", genOk, ref passed, ref failed);

        genOk = true;
        var adxBatch = TacudaBatch.Adx(batchHighs, batchLows, batchInputs, 14);
        for (int s = 0; s < NumSeries && genOk; s++)
        {
            var single = new float[SeriesLen];
            NativeMethods.ct_adx(batchHighs[s], batchLows[s], batchInputs[s], single, SeriesLen, 14);
            genOk = SeriesMatch(adxBatch[s], single, 1e-3f);
        }
        TestResult("TacudaBatch.Adx matches ct_adx", genOk, ref passed, ref failed);

        genOk = true;
        var mfiBatch = TacudaBatch.Mfi(batchHighs, batchLows, batchInputs, batchVolumes, 14);
        for (int s = 0; s < NumSeries && genOk; s++)
        {
            var single = new float[SeriesLen];
            NativeMethods.ct_mfi(batchHighs[s], batchLows[s], batchInputs[s], batchVolumes[s], single, SeriesLen, 14);
            genOk = SeriesMatch(mfiBatch[s], single, 1e-3f);
        }
        TestResult("TacudaBatch.Mfi matches ct_mfi", genOk, ref passed, ref failed);

        var (extMacd, extSignal, _) = TacudaBatch.Macdext(batchInputs, 12, 1, 26, 1, 9, 1);
        TestResult("TacudaBatch.Macdext row 0 == ct_macd(EMA)",
                   SeriesMatch(extMacd[0], macdSingle) && SeriesMatch(extSignal[0], signalSingle),
                   ref passed, ref failed);

        var (aroonDown, aroonUp) = TacudaBatch.Aroon(batchHighs, batchLows, 14);
        bool aroonOk = aroonDown.Length == NumSeries && aroonUp.Length == NumSeries;
        foreach (var rows in new[] { aroonDown, aroonUp })
        {
            foreach (var r in rows)
            {
                foreach (var v in r)
                {
                    if (!float.IsNaN(v) && (v < 0f || v > 100f)) { aroonOk = false; }
                }
            }
        }
        TestResult("TacudaBatch.Aroon bounded [0,100]", aroonOk, ref passed, ref failed);

        // ====================================================================
        // Library metadata
        // ====================================================================
        var version = NativeMethods.ct_version();
        TestResult($"ct_version() == '{version}'",
                   !string.IsNullOrEmpty(version) && version.Split('.').Length == 3,
                   ref passed, ref failed);

        // ====================================================================
        // Cross-validation: SMA(20) should match BBands middle band
        // ====================================================================
        bool smaMatchesBBMiddle = true;
        int validCount = 0;
        for (int i = 0; i < N; i++)
        {
            if (!float.IsNaN(sma[i]) && !float.IsNaN(bbMiddle[i]))
            {
                if (Math.Abs(sma[i] - bbMiddle[i]) > Tolerance)
                {
                    smaMatchesBBMiddle = false;
                    Console.WriteLine($"    SMA/BBMiddle mismatch at {i}: {sma[i]} vs {bbMiddle[i]}");
                    break;
                }
                validCount++;
            }
        }
        TestResult($"Cross-check: SMA(20) == BBands.Middle ({validCount} samples)", 
                   smaMatchesBBMiddle && validCount > 0, ref passed, ref failed);
        
        // ====================================================================
        // Results
        // ====================================================================
        Console.WriteLine();
        Console.WriteLine("=" + new string('=', 69));
        Console.WriteLine($"PIPELINE RESULTS: {passed} passed, {failed} failed");
        Console.WriteLine("=" + new string('=', 69));
        
        return failed > 0 ? 1 : 0;
    }
    
    private static bool SeriesMatch(float[] a, float[] b, float tol = 1e-5f)
    {
        if (a.Length != b.Length)
        {
            return false;
        }
        for (int i = 0; i < a.Length; i++)
        {
            if (float.IsNaN(a[i]) != float.IsNaN(b[i]))
            {
                return false;
            }
            if (!float.IsNaN(a[i]) && Math.Abs(a[i] - b[i]) > tol)
            {
                return false;
            }
        }
        return true;
    }

    private static void TestResult(string name, bool success, ref int passed, ref int failed)
    {
        if (success)
        {
            Console.WriteLine($"  [PASS] {name}");
            passed++;
        }
        else
        {
            Console.WriteLine($"  [FAIL] {name}");
            failed++;
        }
    }
    
    private static bool ValidateRSI(float[] arr)
    {
        foreach (var v in arr)
        {
            if (!float.IsNaN(v) && (v < 0 || v > 100))
                return false;
        }
        return true;
    }
    
    private static bool ValidateNonNegative(float[] arr, string name)
    {
        foreach (var v in arr)
        {
            if (!float.IsNaN(v) && v < 0)
            {
                Console.WriteLine($"    {name} has negative value: {v}");
                return false;
            }
        }
        return true;
    }
    
    private static bool ValidateInRange(float[] arr, float min, float max, string name)
    {
        foreach (var v in arr)
        {
            if (!float.IsNaN(v) && (v < min || v > max))
            {
                Console.WriteLine($"    {name} out of range [{min},{max}]: {v}");
                return false;
            }
        }
        return true;
    }
    
    private static bool ValidateBBands(float[] upper, float[] middle, float[] lower)
    {
        for (int i = 0; i < upper.Length; i++)
        {
            if (!float.IsNaN(upper[i]) && !float.IsNaN(middle[i]) && !float.IsNaN(lower[i]))
            {
                if (upper[i] < middle[i] || middle[i] < lower[i])
                {
                    Console.WriteLine($"    BBands order violated at {i}: U={upper[i]}, M={middle[i]}, L={lower[i]}");
                    return false;
                }
            }
        }
        return true;
    }
    
    private static (float[], float[], float[], float[], float[]) GenerateOHLCV(int size)
    {
        var open = new float[size];
        var high = new float[size];
        var low = new float[size];
        var close = new float[size];
        var volume = new float[size];
        
        for (int i = 0; i < size; i++)
        {
            float t = i;
            
            // Deterministic synthetic price series (matches C++ test)
            float basePrice = 100.0f + 2.0f * MathF.Sin(0.01f * t) + 0.5f * MathF.Cos(0.005f * t) + 0.0001f * t;
            
            float o = basePrice + 0.1f * MathF.Sin(0.03f * t);
            float c = basePrice + 0.1f * MathF.Cos(0.02f * t);
            
            float h = Math.Max(o, c) + 0.2f + 0.3f * MathF.Abs(MathF.Sin(0.01f * t));
            float l = Math.Min(o, c) - 0.2f - 0.3f * MathF.Abs(MathF.Cos(0.01f * t));
            
            open[i] = o;
            high[i] = h;
            low[i] = l;
            close[i] = c;
            
            // Positive, varying volume
            volume[i] = 100000.0f + ((i * 37) % 1000) * 10.0f;
        }
        
        return (open, high, low, close, volume);
    }
}

