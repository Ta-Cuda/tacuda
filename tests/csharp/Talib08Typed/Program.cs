// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System.Text.Json;
using Tacuda.Bindings;

using var document = JsonDocument.Parse(File.ReadAllText(args[0]));
var f = document.RootElement;
float[] Read(string name) => f.GetProperty(name).EnumerateArray().Select(x => x.GetSingle()).ToArray();
void Check(float[] actual, JsonElement expected) {
    if (actual.Length != expected.GetArrayLength()) throw new Exception("length");
    for (int i=0;i<actual.Length;i++) {
        if (expected[i].ValueKind == JsonValueKind.Null) { if (!float.IsNaN(actual[i])) throw new Exception($"NaN at {i}"); }
        else if (!float.IsFinite(actual[i]) || Math.Abs(actual[i]-expected[i].GetDouble()) > 2e-5*Math.Max(1,Math.Abs(expected[i].GetDouble()))) throw new Exception($"value at {i}");
    }
}
void CheckInt(int[] actual, JsonElement expected) {
    if (!actual.SequenceEqual(expected.EnumerateArray().Select(x=>x.GetInt32()))) throw new Exception("integer output mismatch");
}
var o=Read("open");var h=Read("high");var l=Read("low");var c=Read("close");int n=c.Length;
var ho=new float[n];var hh=new float[n];var hl=new float[n];var hc=new float[n];
NativeMethods.ThrowOnError(NativeMethods.ct_ha(o,h,l,c,ho,hh,hl,hc,n),"ha");
Check(ho,f.GetProperty("ha").GetProperty("haOpen"));Check(hh,f.GetProperty("ha").GetProperty("haHigh"));
Check(hl,f.GetProperty("ha").GetProperty("haLow"));Check(hc,f.GetProperty("ha").GetProperty("haClose"));
var up=new int[n];var down=new int[n];
NativeMethods.ThrowOnError(NativeMethods.ct_fractal(h,l,up,down,n,2,2),"fractal");
CheckInt(up,f.GetProperty("fractal").GetProperty("swingHigh"));CheckInt(down,f.GetProperty("fractal").GetProperty("swingLow"));
var trend=TacudaBatch.Supertrend(new[]{h,h},new[]{l,l},new[]{c,c},5,1.5);
for(int row=0;row<2;row++) { Check(trend.Supertrend[row],f.GetProperty("supertrend").GetProperty("real"));CheckInt(trend.Trend[row],f.GetProperty("supertrend").GetProperty("trend")); }
Console.WriteLine("PASS C#: four outputs, int32 flags, mixed batch outputs and double options");
