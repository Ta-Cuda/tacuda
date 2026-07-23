// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;

namespace Tacuda.Bindings;

/// <summary>
/// Columnar OHLCV container for interop with the native TACUDA API.
/// </summary>
public sealed class OhlcvSeries
{
    public int Length { get; }

    public float[] Open { get; }

    public float[] High { get; }

    public float[] Low { get; }

    public float[] Close { get; }

    public float[] Volume { get; }

    public OhlcvSeries(int length)
    {
        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative");
        }

        Length = length;
        Open = new float[length];
        High = new float[length];
        Low = new float[length];
        Close = new float[length];
        Volume = new float[length];
    }

    public OhlcvSeries(float[] open, float[] high, float[] low, float[] close, float[]? volume = null)
    {
        if (open is null)
        {
            throw new ArgumentNullException(nameof(open));
        }
        if (high is null)
        {
            throw new ArgumentNullException(nameof(high));
        }
        if (low is null)
        {
            throw new ArgumentNullException(nameof(low));
        }
        if (close is null)
        {
            throw new ArgumentNullException(nameof(close));
        }

        int length = open.Length;
        EnsureSameLength(length, high.Length, nameof(high));
        EnsureSameLength(length, low.Length, nameof(low));
        EnsureSameLength(length, close.Length, nameof(close));
        if (volume is not null && volume.Length != length)
        {
            throw new ArgumentException("Volume column must match other columns", nameof(volume));
        }

        Length = length;
        Open = CloneArray(open);
        High = CloneArray(high);
        Low = CloneArray(low);
        Close = CloneArray(close);
        Volume = volume is null ? new float[length] : CloneArray(volume);
    }

    public void Set(int index, float open, float high, float low, float close, float volume = 0f)
    {
        if ((uint)index >= (uint)Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        Open[index] = open;
        High[index] = high;
        Low[index] = low;
        Close[index] = close;
        Volume[index] = volume;
    }

    public float[] ToColumnMajor(bool includeVolume = true)
    {
        var result = new float[(includeVolume ? 5 : 4) * Length];
        CopyColumnMajor(result, includeVolume);
        return result;
    }

    public void CopyColumnMajor(float[] destination, bool includeVolume = true)
    {
        if (destination is null)
        {
            throw new ArgumentNullException(nameof(destination));
        }

        int columns = includeVolume ? 5 : 4;
        if (destination.Length < columns * Length)
        {
            throw new ArgumentException("Destination array is too small", nameof(destination));
        }

        Array.Copy(Open, 0, destination, 0, Length);
        Array.Copy(High, 0, destination, Length, Length);
        Array.Copy(Low, 0, destination, 2 * Length, Length);
        Array.Copy(Close, 0, destination, 3 * Length, Length);
        if (includeVolume)
        {
            Array.Copy(Volume, 0, destination, 4 * Length, Length);
        }
    }

    public static OhlcvSeries FromColumnMajor(float[] data, int length, bool includeVolume = true)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }
        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        int columns = includeVolume ? 5 : 4;
        if (data.Length < columns * length)
        {
            throw new ArgumentException("Input array is too small for the requested length", nameof(data));
        }

        var series = new OhlcvSeries(length);
        Array.Copy(data, 0, series.Open, 0, length);
        Array.Copy(data, length, series.High, 0, length);
        Array.Copy(data, 2 * length, series.Low, 0, length);
        Array.Copy(data, 3 * length, series.Close, 0, length);
        if (includeVolume)
        {
            Array.Copy(data, 4 * length, series.Volume, 0, length);
        }
        return series;
    }

    public static OhlcvSeries FromRows(float[] rows, int length, int stride = 5)
    {
        if (rows is null)
        {
            throw new ArgumentNullException(nameof(rows));
        }
        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }
        if (stride != 4 && stride != 5)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be 4 (OHLC) or 5 (OHLCV)");
        }
        if (rows.Length < stride * length)
        {
            throw new ArgumentException("Row array is too small for the requested length", nameof(rows));
        }

        var series = new OhlcvSeries(length);
        for (int i = 0; i < length; ++i)
        {
            int offset = i * stride;
            series.Open[i] = rows[offset];
            series.High[i] = rows[offset + 1];
            series.Low[i] = rows[offset + 2];
            series.Close[i] = rows[offset + 3];
            series.Volume[i] = stride == 5 ? rows[offset + 4] : 0f;
        }
        return series;
    }

    private static void EnsureSameLength(int expected, int actual, string name)
    {
        if (actual != expected)
        {
            throw new ArgumentException("Column lengths must match", name);
        }
    }

    private static float[] CloneArray(float[] source)
    {
        var clone = new float[source.Length];
        Array.Copy(source, clone, source.Length);
        return clone;
    }
}
