// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;
using Tacuda.Bindings;
using Tacuda.Tests;

// Check command line args
if (args.Length > 0 && args[0] == "--test")
{
    return TestBindings.RunAllTests();
}

if (args.Length > 0 && args[0] == "--pipeline")
{
    return TestPipeline.RunPipelineTest();
}

// Original demo code
int length = 1024;
var input = new float[length];
for (int i = 0; i < input.Length; ++i)
{
    input[i] = (float)Math.Sin(0.01 * i);
}

var smaOutput = new float[length];
int status = NativeMethods.ct_sma(input, smaOutput, length, period: 14);
if (status != 0)
{
    throw new InvalidOperationException($"ct_sma failed with status {status}");
}

Console.WriteLine($"SMA[0..4]: {string.Join(", ", smaOutput[0], smaOutput[1], smaOutput[2], smaOutput[3], smaOutput[4])}");

// Demonstrate the OHLCV container working with the native bindings.
var open = new float[] { 1.0f, 2.0f, 1.5f, 1.8f, 2.1f };
var high = new float[] { 1.2f, 2.3f, 1.6f, 2.0f, 2.4f };
var low = new float[] { 0.9f, 1.9f, 1.3f, 1.7f, 1.8f };
var close = new float[] { 1.1f, 2.1f, 1.4f, 1.9f, 2.2f };
var candles = new OhlcvSeries(open, high, low, close);
for (int i = 0; i < candles.Length; ++i)
{
    candles.Volume[i] = 1000f + 50f * i;
}

var imiOutput = new float[candles.Length];
status = NativeMethods.ct_imi(candles.Open, candles.Close, imiOutput, candles.Length, period: 3);
if (status != 0)
{
    throw new InvalidOperationException($"ct_imi failed with status {status}");
}

Console.WriteLine($"IMI[0..{candles.Length - 1}]: {string.Join(", ", imiOutput)}");

return 0;
