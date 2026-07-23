// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;

namespace Tacuda.Bindings;

/// <summary>
/// Structured-input entry points: run indicators directly on an
/// <see cref="OhlcvSeries"/> without unpacking columns manually.
/// </summary>
public static class OhlcvSeriesExtensions
{
    private static float[] Output(OhlcvSeries s) => new float[s.Length];

    public static float[] Atr(this OhlcvSeries s, int period = 14, float initial = 0f)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_atr(s.High, s.Low, s.Close, output, s.Length, period, initial),
            nameof(NativeMethods.ct_atr));
        return output;
    }

    public static float[] Natr(this OhlcvSeries s, int period = 14)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_natr(s.High, s.Low, s.Close, output, s.Length, period),
            nameof(NativeMethods.ct_natr));
        return output;
    }

    public static float[] TrueRange(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_trange(s.High, s.Low, s.Close, output, s.Length),
            nameof(NativeMethods.ct_trange));
        return output;
    }

    public static float[] Adx(this OhlcvSeries s, int period = 14)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_adx(s.High, s.Low, s.Close, output, s.Length, period),
            nameof(NativeMethods.ct_adx));
        return output;
    }

    public static float[] Cci(this OhlcvSeries s, int period = 14)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_cci(s.High, s.Low, s.Close, output, s.Length, period),
            nameof(NativeMethods.ct_cci));
        return output;
    }

    public static float[] Willr(this OhlcvSeries s, int period = 14)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_willr(s.High, s.Low, s.Close, output, s.Length, period),
            nameof(NativeMethods.ct_willr));
        return output;
    }

    public static (float[] K, float[] D) Stochastic(this OhlcvSeries s, int kPeriod = 14, int dPeriod = 3)
    {
        var k = Output(s);
        var d = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_stochastic(s.High, s.Low, s.Close, k, d, s.Length, kPeriod, dPeriod),
            nameof(NativeMethods.ct_stochastic));
        return (k, d);
    }

    public static float[] Ultosc(this OhlcvSeries s, int shortPeriod = 7, int mediumPeriod = 14, int longPeriod = 28)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_ultosc(s.High, s.Low, s.Close, output, s.Length, shortPeriod, mediumPeriod, longPeriod),
            nameof(NativeMethods.ct_ultosc));
        return output;
    }

    public static float[] Mfi(this OhlcvSeries s, int period = 14)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_mfi(s.High, s.Low, s.Close, s.Volume, output, s.Length, period),
            nameof(NativeMethods.ct_mfi));
        return output;
    }

    public static float[] Ad(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_ad(s.High, s.Low, s.Close, s.Volume, output, s.Length),
            nameof(NativeMethods.ct_ad));
        return output;
    }

    public static float[] Obv(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_obv(s.Close, s.Volume, output, s.Length),
            nameof(NativeMethods.ct_obv));
        return output;
    }

    public static float[] Sar(this OhlcvSeries s, float step = 0.02f, float maxAcceleration = 0.2f)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_sar(s.High, s.Low, output, s.Length, step, maxAcceleration),
            nameof(NativeMethods.ct_sar));
        return output;
    }

    public static float[] AvgPrice(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_avgprice(s.Open, s.High, s.Low, s.Close, output, s.Length),
            nameof(NativeMethods.ct_avgprice));
        return output;
    }

    public static float[] Bop(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_bop(s.Open, s.High, s.Low, s.Close, output, s.Length),
            nameof(NativeMethods.ct_bop));
        return output;
    }

    public static float[] CdlDoji(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_cdl_doji(s.Open, s.High, s.Low, s.Close, output, s.Length),
            nameof(NativeMethods.ct_cdl_doji));
        return output;
    }

    public static float[] CdlEngulfing(this OhlcvSeries s)
    {
        var output = Output(s);
        NativeMethods.ThrowOnError(
            NativeMethods.ct_cdl_engulfing(s.Open, s.High, s.Low, s.Close, output, s.Length),
            nameof(NativeMethods.ct_cdl_engulfing));
        return output;
    }
}
