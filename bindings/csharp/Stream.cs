// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;
using System.Runtime.InteropServices;

namespace Tacuda.Bindings;

/// <summary>
/// Streaming (incremental update) API — see docs/STREAMING_API.md.
/// A stream carries indicator state across calls, so a newly closed bar costs
/// a state update instead of a full-series recompute.
/// </summary>
/// <remarks>
/// Hand-written rather than generated: the generator models array-in/array-out
/// calls, while a stream is a handle with a lifetime. Handles are not
/// thread-safe; use one per producer thread or synchronize externally.
/// </remarks>
public static partial class NativeMethods
{
    // --- shared lifetime / introspection ------------------------------------

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_destroy(IntPtr stream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_reset(IntPtr stream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_lookback(IntPtr stream, out int outLookback);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_bars(IntPtr stream, out long outBars);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_outputs(IntPtr stream, out int outCount);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_destroy(IntPtr batch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_reset(IntPtr batch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_lookback(IntPtr batch, out int outLookback);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_bars(IntPtr batch, out long outBars);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_outputs(IntPtr batch, out int outCount);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_series(IntPtr batch, out int outSeries);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_stream_batch_set_exec(IntPtr batch, int exec);

    // --- create -------------------------------------------------------------

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_sma_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ema_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_wma_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_rsi_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_atr_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_natr_stream_create(int period, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_obv_stream_create(out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ad_stream_create(out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_macd_stream_create(int fastPeriod, int slowPeriod, int signalPeriod, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_bbands_stream_create(int period, float upperMul, float lowerMul, out IntPtr outStream);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_sma_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ema_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_wma_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_rsi_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_atr_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_natr_stream_batch_create(int period, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_obv_stream_batch_create(int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ad_stream_batch_create(int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_macd_stream_batch_create(int fastPeriod, int slowPeriod, int signalPeriod, int numSeries, out IntPtr outBatch);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_bbands_stream_batch_create(int period, float upperMul, float lowerMul, int numSeries, out IntPtr outBatch);

    // --- append -------------------------------------------------------------

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_sma_stream_append(IntPtr stream, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ema_stream_append(IntPtr stream, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_wma_stream_append(IntPtr stream, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_rsi_stream_append(IntPtr stream, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_atr_stream_append(IntPtr stream, float[] high, float[] low, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_natr_stream_append(IntPtr stream, float[] high, float[] low, float[] close, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_obv_stream_append(IntPtr stream, float[] close, float[] volume, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ad_stream_append(IntPtr stream, float[] high, float[] low, float[] close, float[] volume, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_macd_stream_append(IntPtr stream, float[] close, int n, float[] outMacd, float[] outSignal, float[] outHist);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_bbands_stream_append(IntPtr stream, float[] close, int n, float[] outUpper, float[] outMiddle, float[] outLower);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_sma_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ema_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_wma_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_rsi_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_atr_stream_batch_append(IntPtr batch, float[] highs, float[] lows, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_natr_stream_batch_append(IntPtr batch, float[] highs, float[] lows, float[] closes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_obv_stream_batch_append(IntPtr batch, float[] closes, float[] volumes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_ad_stream_batch_append(IntPtr batch, float[] highs, float[] lows, float[] closes, float[] volumes, int n, float[] outValues);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_macd_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outMacd, float[] outSignal, float[] outHist);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int ct_bbands_stream_batch_append(IntPtr batch, float[] closes, int n, float[] outUpper, float[] outMiddle, float[] outLower);
}

/// <summary>Where a batch append runs; see docs/STREAMING_API.md for the measured crossover.</summary>
public enum StreamExec
{
    /// <summary>CPU below ~2048 series, GPU above it.</summary>
    Auto = 0,
    Cpu = 1,
    Gpu = 2,
}

/// <summary>
/// Base for every stream: owns the native handle and the calls that do not
/// depend on the indicator. Dispose (or <c>using</c>) releases it; using a
/// disposed stream throws rather than reaching native code.
/// </summary>
public abstract class TacudaStream : IDisposable
{
    private protected IntPtr Handle;
    private readonly bool _isBatch;

    private protected TacudaStream(IntPtr handle, bool isBatch)
    {
        Handle = handle;
        _isBatch = isBatch;
    }

    private protected IntPtr Require()
    {
        if (Handle == IntPtr.Zero)
        {
            throw new ObjectDisposedException(GetType().Name);
        }
        return Handle;
    }

    /// <summary>Drop state, keep parameters — equivalent to a fresh stream.</summary>
    public void Reset()
    {
        int rc = _isBatch
            ? NativeMethods.ct_stream_batch_reset(Require())
            : NativeMethods.ct_stream_reset(Require());
        NativeMethods.ThrowOnError(rc, "ct_stream_reset");
    }

    /// <summary>Bars needed before the first non-NaN output.</summary>
    public int Lookback
    {
        get
        {
            int rc = _isBatch
                ? NativeMethods.ct_stream_batch_lookback(Require(), out int value)
                : NativeMethods.ct_stream_lookback(Require(), out value);
            NativeMethods.ThrowOnError(rc, "ct_stream_lookback");
            return value;
        }
    }

    /// <summary>Bars appended since creation or the last reset.</summary>
    public long Bars
    {
        get
        {
            int rc = _isBatch
                ? NativeMethods.ct_stream_batch_bars(Require(), out long value)
                : NativeMethods.ct_stream_bars(Require(), out value);
            NativeMethods.ThrowOnError(rc, "ct_stream_bars");
            return value;
        }
    }

    /// <summary>Number of output lines the matching Append writes.</summary>
    public int Outputs
    {
        get
        {
            int rc = _isBatch
                ? NativeMethods.ct_stream_batch_outputs(Require(), out int value)
                : NativeMethods.ct_stream_outputs(Require(), out value);
            NativeMethods.ThrowOnError(rc, "ct_stream_outputs");
            return value;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (Handle != IntPtr.Zero)
        {
            if (_isBatch)
            {
                NativeMethods.ct_stream_batch_destroy(Handle);
            }
            else
            {
                NativeMethods.ct_stream_destroy(Handle);
            }
            Handle = IntPtr.Zero;
        }
    }

    ~TacudaStream() => Dispose(false);

    private protected static float[] Allocate(int n) => new float[n];

    private protected static void RequireBars(int n)
    {
        if (n <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "need at least one bar");
        }
    }

    private protected static void RequireSameLength(string what, params float[][] arrays)
    {
        for (int i = 1; i < arrays.Length; ++i)
        {
            if (arrays[i].Length != arrays[0].Length)
            {
                throw new ArgumentException(
                    $"{what}: inputs must have equal length ({arrays[0].Length} vs {arrays[i].Length})");
            }
        }
    }
}

/// <summary>Base for portfolio streams: adds series count and execution policy.</summary>
public abstract class TacudaStreamBatch : TacudaStream
{
    private protected TacudaStreamBatch(IntPtr handle) : base(handle, true) { }

    /// <summary>Series carried by this handle.</summary>
    public int SeriesCount
    {
        get
        {
            NativeMethods.ThrowOnError(
                NativeMethods.ct_stream_batch_series(Require(), out int value),
                "ct_stream_batch_series");
            return value;
        }
    }

    /// <summary>Pick where appends run.</summary>
    public void SetExec(StreamExec exec)
    {
        NativeMethods.ThrowOnError(
            NativeMethods.ct_stream_batch_set_exec(Require(), (int)exec),
            "ct_stream_batch_set_exec");
    }

    /// <summary>Bars per series in a series-major matrix of `total` elements.</summary>
    private protected int BarsPerSeries(int total)
    {
        int series = SeriesCount;
        if (total % series != 0)
        {
            throw new ArgumentException(
                $"input length {total} is not a multiple of the {series} series in this batch");
        }
        int n = total / series;
        RequireBars(n);
        return n;
    }
}
