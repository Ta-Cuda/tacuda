// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;
using System.Diagnostics;
using System.Linq;
using Tacuda.Bindings;

namespace Tacuda.Tests;

/// <summary>
/// Test suite for TACUDA C# bindings
/// </summary>
public static class TestBindings
{
    private const float Tolerance = 1e-5f;
    
    public static int RunAllTests()
    {
        Console.WriteLine("=" + new string('=', 69));
        Console.WriteLine("TACUDA C# Bindings Test Suite");
        Console.WriteLine("=" + new string('=', 69));
        
        int passed = 0;
        int failed = 0;
        
        // Moving Averages
        RunTest("SMA", TestSMA, ref passed, ref failed);
        RunTest("EMA", TestEMA, ref passed, ref failed);
        RunTest("WMA", TestWMA, ref passed, ref failed);
        RunTest("DEMA", TestDEMA, ref passed, ref failed);
        RunTest("TEMA", TestTEMA, ref passed, ref failed);
        
        // Momentum
        RunTest("RSI", TestRSI, ref passed, ref failed);
        RunTest("Momentum", TestMomentum, ref passed, ref failed);
        RunTest("ROC", TestROC, ref passed, ref failed);
        
        // Volatility
        RunTest("StdDev", TestStdDev, ref passed, ref failed);
        RunTest("ATR", TestATR, ref passed, ref failed);
        RunTest("BBands", TestBBands, ref passed, ref failed);
        
        // Oscillators
        RunTest("Stochastic", TestStochastic, ref passed, ref failed);
        RunTest("WillR", TestWillR, ref passed, ref failed);
        RunTest("CCI", TestCCI, ref passed, ref failed);
        
        // Volume
        RunTest("AD", TestAD, ref passed, ref failed);
        RunTest("OBV", TestOBV, ref passed, ref failed);
        RunTest("MFI", TestMFI, ref passed, ref failed);
        
        // Price Transform
        RunTest("AvgPrice", TestAvgPrice, ref passed, ref failed);
        RunTest("TypPrice", TestTypPrice, ref passed, ref failed);
        
        // Statistical
        RunTest("Min/Max", TestMinMax, ref passed, ref failed);
        RunTest("Sum", TestSum, ref passed, ref failed);
        RunTest("LinearReg", TestLinearReg, ref passed, ref failed);
        
        // Additional Moving Averages
        RunTest("TRIMA", TestTrima, ref passed, ref failed);
        RunTest("T3", TestT3, ref passed, ref failed);
        RunTest("KAMA", TestKama, ref passed, ref failed);
        RunTest("MA_Generic", TestMA, ref passed, ref failed);
        
        // Additional Momentum
        RunTest("Change", TestChange, ref passed, ref failed);
        RunTest("ROCP", TestRocp, ref passed, ref failed);
        RunTest("ROCR", TestRocr, ref passed, ref failed);
        RunTest("ROCR100", TestRocr100, ref passed, ref failed);
        RunTest("TRIX", TestTrix, ref passed, ref failed);
        RunTest("CMO", TestCmo, ref passed, ref failed);
        RunTest("APO", TestApo, ref passed, ref failed);
        RunTest("PPO", TestPpo, ref passed, ref failed);
        RunTest("PVO", TestPvo, ref passed, ref failed);
        
        // MACD family
        RunTest("MACD_LINE", TestMacdLine, ref passed, ref failed);
        RunTest("MACD", TestMacd, ref passed, ref failed);
        RunTest("MACDFIX", TestMacdfix, ref passed, ref failed);
        RunTest("MACDEXT", TestMacdext, ref passed, ref failed);
        RunTest("MAMA", TestMama, ref passed, ref failed);
        
        // Vector math (element-wise)
        RunTest("VectorMath_Unary", TestVectorMathUnary, ref passed, ref failed);
        RunTest("VectorMath_Binary", TestVectorMathBinary, ref passed, ref failed);
        
        // Device API
        RunTest("Device_Pipeline", TestDevicePipeline, ref passed, ref failed);
        RunTest("Device_VectorMath", TestDeviceVectorMath, ref passed, ref failed);
        RunTest("Device_BufferPool", TestDeviceBufferPool, ref passed, ref failed);
        
        // Statistics
        RunTest("AvgDev", TestAvgDev, ref passed, ref failed);
        RunTest("Variance", TestVariance, ref passed, ref failed);
        RunTest("MaxIndex", TestMaxIndex, ref passed, ref failed);
        RunTest("MinIndex", TestMinIndex, ref passed, ref failed);
        RunTest("MinMaxIndex", TestMinMaxIndex, ref passed, ref failed);
        RunTest("Beta", TestBeta, ref passed, ref failed);
        RunTest("Correl", TestCorrel, ref passed, ref failed);
        RunTest("Midpoint", TestMidpoint, ref passed, ref failed);
        
        // Linear Regression family
        RunTest("LinearReg_Slope", TestLinearRegSlope, ref passed, ref failed);
        RunTest("LinearReg_Intercept", TestLinearRegIntercept, ref passed, ref failed);
        RunTest("LinearReg_Angle", TestLinearRegAngle, ref passed, ref failed);
        RunTest("TSF", TestTsf, ref passed, ref failed);
        
        // Volatility
        RunTest("NATR", TestNatr, ref passed, ref failed);
        RunTest("TRANGE", TestTrange, ref passed, ref failed);
        RunTest("AccBands", TestAccBands, ref passed, ref failed);
        
        // Oscillators
        RunTest("StochF", TestStochF, ref passed, ref failed);
        RunTest("StochRSI", TestStochRSI, ref passed, ref failed);
        RunTest("UltOsc", TestUltOsc, ref passed, ref failed);
        
        // Directional Movement
        RunTest("ADX", TestADX, ref passed, ref failed);
        RunTest("ADXR", TestADXR, ref passed, ref failed);
        RunTest("DX", TestDX, ref passed, ref failed);
        RunTest("PLUS_DM", TestPlusDM, ref passed, ref failed);
        RunTest("MINUS_DM", TestMinusDM, ref passed, ref failed);
        RunTest("PLUS_DI", TestPlusDI, ref passed, ref failed);
        RunTest("MINUS_DI", TestMinusDI, ref passed, ref failed);
        
        // Volume
        RunTest("ADOSC", TestAdosc, ref passed, ref failed);
        RunTest("IMI", TestIMI, ref passed, ref failed);
        RunTest("NVI", TestNVI, ref passed, ref failed);
        RunTest("PVI", TestPVI, ref passed, ref failed);
        
        // SAR
        RunTest("SAR", TestSar, ref passed, ref failed);
        RunTest("SAREXT", TestSarext, ref passed, ref failed);
        
        // Aroon
        RunTest("Aroon", TestAroon, ref passed, ref failed);
        RunTest("AroonOsc", TestAroonOsc, ref passed, ref failed);
        
        // Price Transform
        RunTest("MedPrice", TestMedPrice, ref passed, ref failed);
        RunTest("WclPrice", TestWclPrice, ref passed, ref failed);
        RunTest("MidPrice", TestMidPrice, ref passed, ref failed);
        RunTest("BOP", TestBop, ref passed, ref failed);
        
        // Hilbert Transform
        RunTest("HT_DCPERIOD", TestHtDcperiod, ref passed, ref failed);
        RunTest("HT_DCPHASE", TestHtDcphase, ref passed, ref failed);
        RunTest("HT_TRENDLINE", TestHtTrendline, ref passed, ref failed);
        RunTest("HT_TRENDMODE", TestHtTrendmode, ref passed, ref failed);
        RunTest("HT_PHASOR", TestHtPhasor, ref passed, ref failed);
        RunTest("HT_SINE", TestHtSine, ref passed, ref failed);
        
        // Candlestick Patterns
        RunTest("CDL_Doji", TestCdlDoji, ref passed, ref failed);
        RunTest("CDL_Hammer", TestCdlHammer, ref passed, ref failed);
        RunTest("CDL_Engulfing", TestCdlEngulfing, ref passed, ref failed);
        RunTest("CDL_ThreeWhiteSoldiers", TestCdlThreeWhiteSoldiers, ref passed, ref failed);
        RunTest("CDL_ThreeBlackCrows", TestCdlThreeBlackCrows, ref passed, ref failed);
        RunTest("CDL_MorningStar", TestCdlMorningStar, ref passed, ref failed);
        RunTest("CDL_EveningStar", TestCdlEveningStar, ref passed, ref failed);
        RunTest("CDL_Harami", TestCdlHarami, ref passed, ref failed);
        RunTest("CDL_SpinningTop", TestCdlSpinningTop, ref passed, ref failed);
        RunTest("CDL_Marubozu", TestCdlMarubozu, ref passed, ref failed);
        
        // Performance
        RunTest("Performance_1M", TestPerformance, ref passed, ref failed);
        
        // OHLCV Container
        RunTest("OhlcvSeries", TestOhlcvSeries, ref passed, ref failed);
        
        Console.WriteLine();
        Console.WriteLine("=" + new string('=', 69));
        Console.WriteLine($"RESULTS: {passed} passed, {failed} failed, {passed + failed} total");
        Console.WriteLine("=" + new string('=', 69));
        
        return failed > 0 ? 1 : 0;
    }
    
    private static void RunTest(string name, Func<bool> test, ref int passed, ref int failed)
    {
        try
        {
            bool result = test();
            if (result)
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
        catch (Exception ex)
        {
            Console.WriteLine($"  [ERROR] {name}: {ex.Message}");
            failed++;
        }
    }
    
    // =========================================================================
    // Moving Averages Tests
    // =========================================================================
    
    private static bool TestSMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        const int period = 10;
        
        int status = NativeMethods.ct_sma(input, output, size, period: period);
        if (status != 0) return false;
        
        // TA-Lib compatible: warmup is at the head (period-1 samples).
        for (int i = 0; i < period - 1; i++)
        {
            if (!float.IsNaN(output[i])) return false;
        }
        
        // Check a known value (manual calculation)
        float expectedSMA = input.Take(period).Average();
        return !float.IsNaN(output[period - 1]) &&
               Math.Abs(output[period - 1] - expectedSMA) < Tolerance;
    }
    
    private static bool TestEMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_ema(input, output, size, period: 10);
        if (status != 0) return false;
        
        // EMA should have values after warmup
        return !float.IsNaN(output[9]) && output[9] != 0;
    }
    
    private static bool TestWMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        const int period = 10;
        
        int status = NativeMethods.ct_wma(input, output, size, period: period);
        if (status != 0) return false;

        // Warmup at head (period-1).
        for (int i = 0; i < period - 1; i++)
        {
            if (!float.IsNaN(output[i])) return false;
        }

        // TA-Lib WMA uses weights 1..period (oldest->newest), output at index period-1.
        float denom = 0.5f * period * (period + 1);
        float weighted = 0f;
        for (int k = 0; k < period; k++)
        {
            weighted += input[k] * (k + 1);
        }
        float expected = weighted / denom;

        return !float.IsNaN(output[period - 1]) &&
               Math.Abs(output[period - 1] - expected) < 1e-4f;
    }
    
    private static bool TestDEMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_dema(input, output, size, period: 10);
        return status == 0;
    }
    
    private static bool TestTEMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_tema(input, output, size, period: 10);
        return status == 0;
    }
    
    // =========================================================================
    // Momentum Tests
    // =========================================================================
    
    private static bool TestRSI()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_rsi(input, output, size, period: 14);
        if (status != 0) return false;
        
        // RSI should be between 0 and 100
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(output[i]) && (output[i] < 0 || output[i] > 100))
                return false;
        }
        return true;
    }
    
    private static bool TestMomentum()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_momentum(input, output, size, period: 10);
        return status == 0;
    }
    
    private static bool TestROC()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_roc(input, output, size, period: 10);
        return status == 0;
    }
    
    // =========================================================================
    // Volatility Tests
    // =========================================================================
    
    private static bool TestStdDev()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_stddev(input, output, size, period: 10);
        if (status != 0) return false;
        
        // StdDev should be non-negative
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(output[i]) && output[i] < 0)
                return false;
        }
        return true;
    }
    
    private static bool TestATR()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_atr(high, low, close, output, size, period: 14, initial: 0f);
        if (status != 0) return false;
        
        // ATR should be non-negative
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(output[i]) && output[i] < 0)
                return false;
        }
        return true;
    }
    
    private static bool TestBBands()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var upper = new float[size];
        var middle = new float[size];
        var lower = new float[size];
        
        int status = NativeMethods.ct_bbands(input, upper, middle, lower, size, 
                                              period: 20, upperMul: 2.0f, lowerMul: 2.0f);
        if (status != 0) return false;
        
        // Upper >= Middle >= Lower
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(upper[i]) && !float.IsNaN(middle[i]) && !float.IsNaN(lower[i]))
            {
                if (upper[i] < middle[i] || middle[i] < lower[i])
                    return false;
            }
        }
        return true;
    }
    
    // =========================================================================
    // Oscillators Tests
    // =========================================================================
    
    private static bool TestStochastic()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var k = new float[size];
        var d = new float[size];
        
        int status = NativeMethods.ct_stochastic(high, low, close, k, d, size, 
                                                  kPeriod: 14, dPeriod: 3);
        if (status != 0) return false;
        
        // K and D should be between 0 and 100
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(k[i]) && (k[i] < 0 || k[i] > 100)) return false;
            if (!float.IsNaN(d[i]) && (d[i] < 0 || d[i] > 100)) return false;
        }
        return true;
    }
    
    private static bool TestWillR()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_willr(high, low, close, output, size, period: 14);
        if (status != 0) return false;
        
        // Williams %R should be between -100 and 0
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(output[i]) && (output[i] < -100 || output[i] > 0))
                return false;
        }
        return true;
    }
    
    private static bool TestCCI()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_cci(high, low, close, output, size, period: 20);
        return status == 0;
    }
    
    // =========================================================================
    // Volume Tests
    // =========================================================================
    
    private static bool TestAD()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_ad(high, low, close, volume, output, size);
        return status == 0;
    }
    
    private static bool TestOBV()
    {
        int size = 100;
        var close = GenerateTestData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_obv(close, volume, output, size);
        return status == 0;
    }
    
    private static bool TestMFI()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_mfi(high, low, close, volume, output, size, period: 14);
        if (status != 0) return false;
        
        // MFI should be between 0 and 100
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(output[i]) && (output[i] < 0 || output[i] > 100))
                return false;
        }
        return true;
    }
    
    // =========================================================================
    // Price Transform Tests
    // =========================================================================
    
    private static bool TestAvgPrice()
    {
        int size = 100;
        var open = GenerateTestData(size);
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_avgprice(open, high, low, close, output, size);
        if (status != 0) return false;
        
        // AvgPrice = (O + H + L + C) / 4
        float expected = (open[0] + high[0] + low[0] + close[0]) / 4.0f;
        return Math.Abs(output[0] - expected) < Tolerance;
    }
    
    private static bool TestTypPrice()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_typprice(high, low, close, output, size);
        if (status != 0) return false;
        
        // TypPrice = (H + L + C) / 3
        float expected = (high[0] + low[0] + close[0]) / 3.0f;
        return Math.Abs(output[0] - expected) < Tolerance;
    }
    
    // =========================================================================
    // Statistical Tests
    // =========================================================================
    
    private static bool TestMinMax()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var minOut = new float[size];
        var maxOut = new float[size];
        
        int status = NativeMethods.ct_minmax(input, minOut, maxOut, size, period: 10);
        if (status != 0) return false;
        
        // Min <= Max
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(minOut[i]) && !float.IsNaN(maxOut[i]))
            {
                if (minOut[i] > maxOut[i]) return false;
            }
        }
        return true;
    }
    
    private static bool TestSum()
    {
        int size = 100;
        var input = new float[size];
        for (int i = 0; i < size; i++) input[i] = 1.0f; // All ones
        var output = new float[size];
        
        int status = NativeMethods.ct_sum(input, output, size, period: 10);
        if (status != 0) return false;
        
        // TA-Lib alignment: period-1 warm-up NaNs, first sum at index 9.
        return float.IsNaN(output[0]) && Math.Abs(output[9] - 10.0f) < Tolerance;
    }
    
    private static bool TestLinearReg()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_linearreg(input, output, size, period: 14);
        return status == 0;
    }
    
    // =========================================================================
    // Additional Moving Averages
    // =========================================================================
    
    private static bool TestTrima()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_trima(input, output, size, period: 10);
        return status == 0 && !float.IsNaN(output[size - 1]);
    }
    
    private static bool TestT3()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_t3(input, output, size, period: 5, vFactor: 0.7f);
        return status == 0;
    }
    
    private static bool TestKama()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_kama(input, output, size, period: 10, fastPeriod: 2, slowPeriod: 30);
        return status == 0 && !float.IsNaN(output[size - 1]);
    }
    
    private static bool TestMA()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_ma(input, output, size, period: 10, type: 0); // SMA
        return status == 0 && !float.IsNaN(output[size - 1]);
    }
    
    // =========================================================================
    // Additional Momentum
    // =========================================================================
    
    private static bool TestChange()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_change(input, output, size, period: 10) == 0;
    }
    
    private static bool TestRocp()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_rocp(input, output, size, period: 10) == 0;
    }
    
    private static bool TestRocr()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_rocr(input, output, size, period: 10) == 0;
    }
    
    private static bool TestRocr100()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_rocr100(input, output, size, period: 10) == 0;
    }
    
    private static bool TestTrix()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_trix(input, output, size, period: 14) == 0;
    }
    
    private static bool TestCmo()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_cmo(input, output, size, period: 14) == 0;
    }
    
    private static bool TestApo()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_apo(input, output, size, fastPeriod: 12, slowPeriod: 26) == 0;
    }
    
    private static bool TestPpo()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_ppo(input, output, size, fastPeriod: 12, slowPeriod: 26) == 0;
    }
    
    private static bool TestPvo()
    {
        int size = 100;
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        return NativeMethods.ct_pvo(volume, output, size, fastPeriod: 12, slowPeriod: 26) == 0;
    }
    
    // =========================================================================
    // MACD family
    // =========================================================================
    
    private static bool TestMacdLine()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_macd_line(input, output, size, fastPeriod: 12, slowPeriod: 26) == 0;
    }
    
    private static bool TestMacd()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var macdOut = new float[size];
        var signal = new float[size];
        var hist = new float[size];
        return NativeMethods.ct_macd(input, macdOut, signal, hist, size, 12, 26, 9, 0) == 0;
    }
    
    private static bool TestMacdfix()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var macdOut = new float[size];
        var signal = new float[size];
        var hist = new float[size];
        return NativeMethods.ct_macdfix(input, macdOut, signal, hist, size, signalPeriod: 9) == 0;
    }
    
    private static bool TestMama()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var mama = new float[size];
        var fama = new float[size];
        return NativeMethods.ct_mama(input, mama, fama, size, fastLimit: 0.5f, slowLimit: 0.05f) == 0;
    }
    
    private static bool TestMacdext()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var extMacd = new float[size];
        var extSignal = new float[size];
        var extHist = new float[size];
        
        // MACDEXT with EMA everywhere must reproduce classic MACD(EMA).
        int status = NativeMethods.ct_macdext(input, extMacd, extSignal, extHist, size,
            fastPeriod: 12, fastType: 1, slowPeriod: 26, slowType: 1, signalPeriod: 9, signalType: 1);
        if (status != 0) return false;
        
        var refMacd = new float[size];
        var refSignal = new float[size];
        var refHist = new float[size];
        status = NativeMethods.ct_macd(input, refMacd, refSignal, refHist, size, 12, 26, 9, 1);
        if (status != 0) return false;
        
        int compared = 0;
        for (int i = 0; i < size; i++)
        {
            if (float.IsNaN(extMacd[i]) || float.IsNaN(refMacd[i])) continue;
            if (Math.Abs(extMacd[i] - refMacd[i]) > 1e-3f) return false;
            compared++;
        }
        if (compared == 0) return false;
        
        // Mixed MA types must at least produce valid values.
        status = NativeMethods.ct_macdext(input, extMacd, extSignal, extHist, size,
            fastPeriod: 12, fastType: 0, slowPeriod: 26, slowType: 1, signalPeriod: 9, signalType: 2);
        return status == 0 && extMacd.Any(x => !float.IsNaN(x));
    }
    
    // =========================================================================
    // Vector Math
    // =========================================================================
    
    private static bool TestVectorMathUnary()
    {
        int size = 1000;
        var input = new float[size];
        for (int i = 0; i < size; i++) input[i] = -0.95f + 1.9f * i / (size - 1);
        var positive = new float[size];
        for (int i = 0; i < size; i++) positive[i] = 0.1f + 0.01f * i;
        var output = new float[size];
        
        var cases = new (string Name, Func<float[], float[], int, IntPtr, int> Fn, float[] Input, Func<double, double> Reference)[]
        {
            ("sin", (a, b, n, s) => NativeMethods.ct_sin(a, b, n, s), input, Math.Sin),
            ("cos", (a, b, n, s) => NativeMethods.ct_cos(a, b, n, s), input, Math.Cos),
            ("tan", (a, b, n, s) => NativeMethods.ct_tan(a, b, n, s), input, Math.Tan),
            ("asin", (a, b, n, s) => NativeMethods.ct_asin(a, b, n, s), input, Math.Asin),
            ("acos", (a, b, n, s) => NativeMethods.ct_acos(a, b, n, s), input, Math.Acos),
            ("atan", (a, b, n, s) => NativeMethods.ct_atan(a, b, n, s), input, Math.Atan),
            ("sinh", (a, b, n, s) => NativeMethods.ct_sinh(a, b, n, s), input, Math.Sinh),
            ("cosh", (a, b, n, s) => NativeMethods.ct_cosh(a, b, n, s), input, Math.Cosh),
            ("tanh", (a, b, n, s) => NativeMethods.ct_tanh(a, b, n, s), input, Math.Tanh),
            ("exp", (a, b, n, s) => NativeMethods.ct_exp(a, b, n, s), input, Math.Exp),
            ("ceil", (a, b, n, s) => NativeMethods.ct_ceil(a, b, n, s), input, Math.Ceiling),
            ("floor", (a, b, n, s) => NativeMethods.ct_floor(a, b, n, s), input, Math.Floor),
            ("ln", (a, b, n, s) => NativeMethods.ct_ln(a, b, n, s), positive, Math.Log),
            ("log10", (a, b, n, s) => NativeMethods.ct_log10(a, b, n, s), positive, Math.Log10),
            ("sqrt", (a, b, n, s) => NativeMethods.ct_sqrt(a, b, n, s), positive, Math.Sqrt),
        };
        
        foreach (var (name, fn, src, reference) in cases)
        {
            if (fn(src, output, size, IntPtr.Zero) != 0)
            {
                Console.WriteLine($"        {name} returned non-zero status");
                return false;
            }
            for (int i = 0; i < size; i += 97)
            {
                float expected = (float)reference(src[i]);
                if (Math.Abs(output[i] - expected) > 1e-4f * Math.Max(1f, Math.Abs(expected)))
                {
                    Console.WriteLine($"        {name}[{i}]: got {output[i]}, expected {expected}");
                    return false;
                }
            }
        }
        return true;
    }
    
    private static bool TestVectorMathBinary()
    {
        int size = 1000;
        var a = new float[size];
        var b = new float[size];
        for (int i = 0; i < size; i++)
        {
            a[i] = 0.1f + 0.01f * i;
            b[i] = -0.95f + 1.9f * i / (size - 1);
        }
        var output = new float[size];
        
        if (NativeMethods.ct_add(a, b, output, size) != 0) return false;
        for (int i = 0; i < size; i += 61)
            if (Math.Abs(output[i] - (a[i] + b[i])) > 1e-4f) return false;
        
        if (NativeMethods.ct_sub(a, b, output, size) != 0) return false;
        for (int i = 0; i < size; i += 61)
            if (Math.Abs(output[i] - (a[i] - b[i])) > 1e-4f) return false;
        
        if (NativeMethods.ct_mult(a, b, output, size) != 0) return false;
        for (int i = 0; i < size; i += 61)
            if (Math.Abs(output[i] - a[i] * b[i]) > 1e-4f) return false;
        
        if (NativeMethods.ct_div(a, b, output, size) != 0) return false;
        for (int i = 0; i < size; i += 61)
        {
            float expected = a[i] / b[i];
            if (float.IsFinite(expected) && Math.Abs(output[i] - expected) > 1e-3f * Math.Max(1f, Math.Abs(expected)))
                return false;
        }
        return true;
    }
    
    // =========================================================================
    // Device API
    // =========================================================================
    
    private static bool TestDevicePipeline()
    {
        int size = 4096;
        var input = GenerateTestData(size);
        
        IntPtr dIn = NativeMethods.ct_device_alloc(size);
        IntPtr dOut = NativeMethods.ct_device_alloc(size);
        if (dIn == IntPtr.Zero || dOut == IntPtr.Zero) return false;
        
        try
        {
            if (NativeMethods.ct_device_upload(dIn, input, size) != 0) return false;
            if (NativeMethods.ct_sma_d(dIn, dOut, size, 20) != 0) return false;
            if (NativeMethods.ct_device_sync() != 0) return false;
            
            var gpu = new float[size];
            if (NativeMethods.ct_device_download(gpu, dOut, size) != 0) return false;
            
            var cpu = new float[size];
            if (NativeMethods.ct_sma(input, cpu, size, 20) != 0) return false;
            
            for (int i = 0; i < size; i += 41)
            {
                if (float.IsNaN(cpu[i]) != float.IsNaN(gpu[i])) return false;
                if (!float.IsNaN(cpu[i]) && Math.Abs(cpu[i] - gpu[i]) > 1e-4f) return false;
            }
            return true;
        }
        finally
        {
            NativeMethods.ct_device_free(dIn);
            NativeMethods.ct_device_free(dOut);
        }
    }
    
    private static bool TestDeviceVectorMath()
    {
        int size = 4096;
        var input = new float[size];
        for (int i = 0; i < size; i++) input[i] = -0.95f + 1.9f * i / (size - 1);
        
        IntPtr dIn = NativeMethods.ct_device_alloc(size);
        IntPtr dOut = NativeMethods.ct_device_alloc(size);
        if (dIn == IntPtr.Zero || dOut == IntPtr.Zero) return false;
        
        try
        {
            if (NativeMethods.ct_device_upload(dIn, input, size) != 0) return false;
            if (NativeMethods.ct_sin_d(dIn, dOut, size) != 0) return false;
            if (NativeMethods.ct_device_sync() != 0) return false;
            
            var gpu = new float[size];
            if (NativeMethods.ct_device_download(gpu, dOut, size) != 0) return false;
            
            for (int i = 0; i < size; i += 37)
            {
                float expected = (float)Math.Sin(input[i]);
                if (Math.Abs(gpu[i] - expected) > 1e-5f) return false;
            }
            return true;
        }
        finally
        {
            NativeMethods.ct_device_free(dIn);
            NativeMethods.ct_device_free(dOut);
        }
    }
    
    private static bool TestDeviceBufferPool()
    {
        // Trigger at least one pooled allocation, then inspect counters.
        int size = 1024;
        var input = GenerateTestData(size);
        var output = new float[size];
        if (NativeMethods.ct_sma(input, output, size, 10) != 0) return false;
        
        ulong count = NativeMethods.ct_device_buffer_pool_allocation_count();
        NativeMethods.ct_device_buffer_pool_cleanup();
        return count < ulong.MaxValue;
    }
    
    // =========================================================================
    // Additional Statistics
    // =========================================================================
    
    private static bool TestAvgDev()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_avgdev(input, output, size, period: 10);
        if (status != 0) return false;
        for (int i = 0; i < size; i++)
            if (!float.IsNaN(output[i]) && output[i] < 0) return false;
        return true;
    }
    
    private static bool TestVariance()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        int status = NativeMethods.ct_var(input, output, size, period: 10);
        if (status != 0) return false;
        for (int i = 0; i < size; i++)
            if (!float.IsNaN(output[i]) && output[i] < 0) return false;
        return true;
    }
    
    private static bool TestMaxIndex()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_maxindex(input, output, size, period: 10) == 0;
    }
    
    private static bool TestMinIndex()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_minindex(input, output, size, period: 10) == 0;
    }
    
    private static bool TestMinMaxIndex()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var minIdx = new float[size];
        var maxIdx = new float[size];
        return NativeMethods.ct_minmaxindex(input, minIdx, maxIdx, size, period: 10) == 0;
    }
    
    private static bool TestBeta()
    {
        int size = 100;
        var x = GenerateTestData(size);
        var y = GenerateTestData(size); // same seed, but that's ok for smoke test
        var output = new float[size];
        return NativeMethods.ct_beta(x, y, output, size, period: 10) == 0;
    }
    
    private static bool TestCorrel()
    {
        int size = 100;
        var x = GenerateTestData(size);
        var y = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_correl(x, y, output, size, period: 10) == 0;
    }
    
    private static bool TestMidpoint()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_midpoint(input, output, size, period: 10) == 0;
    }
    
    // =========================================================================
    // Linear Regression family
    // =========================================================================
    
    private static bool TestLinearRegSlope()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_linearreg_slope(input, output, size, period: 14) == 0;
    }
    
    private static bool TestLinearRegIntercept()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_linearreg_intercept(input, output, size, period: 14) == 0;
    }
    
    private static bool TestLinearRegAngle()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_linearreg_angle(input, output, size, period: 14) == 0;
    }
    
    private static bool TestTsf()
    {
        int size = 100;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_tsf(input, output, size, period: 14) == 0;
    }
    
    // =========================================================================
    // Volatility
    // =========================================================================
    
    private static bool TestNatr()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        int status = NativeMethods.ct_natr(high, low, close, output, size, period: 14);
        if (status != 0) return false;
        for (int i = 0; i < size; i++)
            if (!float.IsNaN(output[i]) && output[i] < 0) return false;
        return true;
    }
    
    private static bool TestTrange()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        int status = NativeMethods.ct_trange(high, low, close, output, size);
        if (status != 0) return false;
        for (int i = 0; i < size; i++)
            if (!float.IsNaN(output[i]) && output[i] < 0) return false;
        return true;
    }
    
    private static bool TestAccBands()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var upper = new float[size];
        var middle = new float[size];
        var lower = new float[size];
        return NativeMethods.ct_accbands(high, low, close, upper, middle, lower, size, period: 20) == 0;
    }
    
    // =========================================================================
    // Oscillators
    // =========================================================================
    
    private static bool TestStochF()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var k = new float[size];
        var d = new float[size];
        int status = NativeMethods.ct_stochf(high, low, close, k, d, size, kPeriod: 5, dPeriod: 3);
        if (status != 0) return false;
        for (int i = 0; i < size; i++)
        {
            if (!float.IsNaN(k[i]) && (k[i] < 0 || k[i] > 100)) return false;
            if (!float.IsNaN(d[i]) && (d[i] < 0 || d[i] > 100)) return false;
        }
        return true;
    }
    
    private static bool TestStochRSI()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var k = new float[size];
        var d = new float[size];
        return NativeMethods.ct_stochrsi(input, k, d, size, rsiPeriod: 14, kPeriod: 5, dPeriod: 3) == 0;
    }
    
    private static bool TestUltOsc()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_ultosc(high, low, close, output, size, 7, 14, 28) == 0;
    }
    
    // =========================================================================
    // Directional Movement
    // =========================================================================
    
    private static bool TestADX()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_adx(high, low, close, output, size, period: 14) == 0;
    }
    
    private static bool TestADXR()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_adxr(high, low, close, output, size, period: 14) == 0;
    }
    
    private static bool TestDX()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_dx(high, low, close, output, size, period: 14) == 0;
    }
    
    private static bool TestPlusDM()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_plus_dm(high, low, output, size, period: 14) == 0;
    }
    
    private static bool TestMinusDM()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_minus_dm(high, low, output, size, period: 14) == 0;
    }
    
    private static bool TestPlusDI()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_plus_di(high, low, close, output, size, period: 14) == 0;
    }
    
    private static bool TestMinusDI()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_minus_di(high, low, close, output, size, period: 14) == 0;
    }
    
    // =========================================================================
    // Volume
    // =========================================================================
    
    private static bool TestAdosc()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        return NativeMethods.ct_adosc(high, low, close, volume, output, size, shortPeriod: 3, longPeriod: 10) == 0;
    }
    
    private static bool TestIMI()
    {
        int size = 100;
        var (open, _, _, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_imi(open, close, output, size, period: 14) == 0;
    }
    
    private static bool TestNVI()
    {
        int size = 100;
        var close = GenerateTestData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        return NativeMethods.ct_nvi(close, volume, output, size) == 0;
    }
    
    private static bool TestPVI()
    {
        int size = 100;
        var close = GenerateTestData(size);
        var volume = GenerateVolumeData(size);
        var output = new float[size];
        return NativeMethods.ct_pvi(close, volume, output, size) == 0;
    }
    
    // =========================================================================
    // SAR
    // =========================================================================
    
    private static bool TestSar()
    {
        int size = 200;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_sar(high, low, output, size, step: 0.02f, maxAcceleration: 0.2f) == 0;
    }
    
    private static bool TestSarext()
    {
        int size = 200;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_sarext(high, low, output, size,
            startValue: 0f, offsetOnReverse: 0f,
            accInitLong: 0.02f, accLong: 0.02f, accMaxLong: 0.2f,
            accInitShort: 0.02f, accShort: 0.02f, accMaxShort: 0.2f) == 0;
    }
    
    // =========================================================================
    // Aroon
    // =========================================================================
    
    private static bool TestAroon()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var up = new float[size];
        var down = new float[size];
        var osc = new float[size];
        return NativeMethods.ct_aroon(high, low, up, down, osc, size, upPeriod: 14, downPeriod: 14) == 0;
    }
    
    private static bool TestAroonOsc()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_aroonosc(high, low, output, size, period: 14) == 0;
    }
    
    // =========================================================================
    // Price Transform
    // =========================================================================
    
    private static bool TestMedPrice()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        int status = NativeMethods.ct_medprice(high, low, output, size);
        if (status != 0) return false;
        float expected = (high[0] + low[0]) / 2.0f;
        return Math.Abs(output[0] - expected) < Tolerance;
    }
    
    private static bool TestWclPrice()
    {
        int size = 100;
        var (high, low, close) = GenerateOHLCData(size);
        var output = new float[size];
        int status = NativeMethods.ct_wclprice(high, low, close, output, size);
        if (status != 0) return false;
        float expected = (high[0] + low[0] + 2 * close[0]) / 4.0f;
        return Math.Abs(output[0] - expected) < Tolerance;
    }
    
    private static bool TestMidPrice()
    {
        int size = 100;
        var (high, low, _) = GenerateOHLCData(size);
        var output = new float[size];
        return NativeMethods.ct_midprice(high, low, output, size, period: 10) == 0;
    }
    
    private static bool TestBop()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_bop(open, high, low, close, output, size) == 0;
    }
    
    // =========================================================================
    // Hilbert Transform
    // =========================================================================
    
    private static bool TestHtDcperiod()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_ht_dcperiod(input, output, size) == 0;
    }
    
    private static bool TestHtDcphase()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_ht_dcphase(input, output, size) == 0;
    }
    
    private static bool TestHtTrendline()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_ht_trendline(input, output, size) == 0;
    }
    
    private static bool TestHtTrendmode()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var output = new float[size];
        return NativeMethods.ct_ht_trendmode(input, output, size) == 0;
    }
    
    private static bool TestHtPhasor()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var inphase = new float[size];
        var quadrature = new float[size];
        return NativeMethods.ct_ht_phasor(input, inphase, quadrature, size) == 0;
    }
    
    private static bool TestHtSine()
    {
        int size = 200;
        var input = GenerateTestData(size);
        var sine = new float[size];
        var leadsine = new float[size];
        return NativeMethods.ct_ht_sine(input, sine, leadsine, size) == 0;
    }
    
    // =========================================================================
    // Candlestick Pattern Tests
    // =========================================================================
    
    private static bool TestCdlDoji()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_cdl_doji(open, high, low, close, output, size);
        return status == 0;
    }
    
    private static bool TestCdlHammer()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_cdl_hammer(open, high, low, close, output, size);
        return status == 0;
    }
    
    private static bool TestCdlEngulfing()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        
        int status = NativeMethods.ct_cdl_engulfing(open, high, low, close, output, size);
        return status == 0;
    }
    
    private static bool TestCdlThreeWhiteSoldiers()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_three_white_soldiers(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlThreeBlackCrows()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_three_black_crows(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlMorningStar()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_morning_star(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlEveningStar()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_evening_star(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlHarami()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_harami(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlSpinningTop()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_spinning_top(open, high, low, close, output, size) == 0;
    }
    
    private static bool TestCdlMarubozu()
    {
        int size = 100;
        var (open, high, low, close) = GenerateOHLCVData(size);
        var output = new float[size];
        return NativeMethods.ct_cdl_marubozu(open, high, low, close, output, size) == 0;
    }
    
    // =========================================================================
    // Performance Test
    // =========================================================================
    
    private static bool TestPerformance()
    {
        int size = 1_000_000;
        var input = GenerateTestData(size);
        var output = new float[size];
        
        var sw = Stopwatch.StartNew();
        int status = NativeMethods.ct_sma(input, output, size, period: 20);
        sw.Stop();
        
        if (status != 0) return false;
        
        Console.WriteLine($"        SMA 1M: {sw.ElapsedMilliseconds}ms");
        
        sw.Restart();
        status = NativeMethods.ct_rsi(input, output, size, period: 14);
        sw.Stop();
        
        if (status != 0) return false;
        
        Console.WriteLine($"        RSI 1M: {sw.ElapsedMilliseconds}ms");
        
        return true;
    }
    
    // =========================================================================
    // OhlcvSeries Container Test
    // =========================================================================
    
    private static bool TestOhlcvSeries()
    {
        // Test construction
        var series = new OhlcvSeries(100);
        if (series.Length != 100) return false;
        
        // Test from columns
        var open = new float[] { 1, 2, 3, 4, 5 };
        var high = new float[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f };
        var low = new float[] { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f };
        var close = new float[] { 1.2f, 2.2f, 3.2f, 4.2f, 5.2f };
        
        var candles = new OhlcvSeries(open, high, low, close);
        if (candles.Length != 5) return false;
        if (Math.Abs(candles.Open[0] - 1.0f) > Tolerance) return false;
        if (Math.Abs(candles.High[2] - 3.5f) > Tolerance) return false;
        
        // Test column major conversion
        var columnMajor = candles.ToColumnMajor(includeVolume: false);
        if (columnMajor.Length != 20) return false; // 4 columns * 5 rows
        
        // Test with native function
        var atrOutput = new float[candles.Length];
        int status = NativeMethods.ct_atr(candles.High, candles.Low, candles.Close, 
                                           atrOutput, candles.Length, period: 3, initial: 0f);
        
        return status == 0;
    }
    
    // =========================================================================
    // Helper Methods
    // =========================================================================
    
    private static float[] GenerateTestData(int size)
    {
        var data = new float[size];
        var random = new Random(42);
        float value = 100.0f;
        
        for (int i = 0; i < size; i++)
        {
            value += (float)(random.NextDouble() - 0.5) * 2;
            data[i] = value;
        }
        return data;
    }
    
    private static (float[] high, float[] low, float[] close) GenerateOHLCData(int size)
    {
        var high = new float[size];
        var low = new float[size];
        var close = new float[size];
        var random = new Random(42);
        float basePrice = 100.0f;
        
        for (int i = 0; i < size; i++)
        {
            float range = (float)random.NextDouble() * 5;
            float mid = basePrice + (float)(random.NextDouble() - 0.5) * 2;
            high[i] = mid + range / 2;
            low[i] = mid - range / 2;
            close[i] = low[i] + (float)random.NextDouble() * range;
            basePrice = close[i];
        }
        return (high, low, close);
    }
    
    private static (float[] open, float[] high, float[] low, float[] close) GenerateOHLCVData(int size)
    {
        var open = new float[size];
        var high = new float[size];
        var low = new float[size];
        var close = new float[size];
        var random = new Random(42);
        float basePrice = 100.0f;
        
        for (int i = 0; i < size; i++)
        {
            float range = (float)random.NextDouble() * 5;
            open[i] = basePrice;
            high[i] = basePrice + range;
            low[i] = basePrice - range * 0.5f;
            close[i] = low[i] + (float)random.NextDouble() * (high[i] - low[i]);
            basePrice = close[i];
        }
        return (open, high, low, close);
    }
    
    private static float[] GenerateVolumeData(int size)
    {
        var data = new float[size];
        var random = new Random(42);
        
        for (int i = 0; i < size; i++)
        {
            data[i] = 1000000 + (float)random.NextDouble() * 500000;
        }
        return data;
    }
}

