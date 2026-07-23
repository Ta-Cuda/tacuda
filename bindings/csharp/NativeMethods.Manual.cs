// ===========================================================================
//  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
//  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
//  SPDX-License-Identifier: Apache-2.0
// ===========================================================================
using System;
using System.Runtime.InteropServices;

namespace Tacuda.Bindings;

/// <summary>
/// Hand-written P/Invoke declarations for entry points the generator skips
/// (functions returning <c>const char*</c>).
/// </summary>
public static partial class NativeMethods
{
    [DllImport(LibraryName, EntryPoint = "ct_version", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr ct_version_native();

    [DllImport(LibraryName, EntryPoint = "ct_status_string", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr ct_status_string_native(int status);

    [DllImport(LibraryName, EntryPoint = "ct_get_last_error", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr ct_get_last_error_native();

    /// <summary>Native library version string (MAJOR.MINOR.PATCH).</summary>
    public static string ct_version()
        => Marshal.PtrToStringAnsi(ct_version_native()) ?? string.Empty;

    /// <summary>Human-readable description for a <c>ctStatus_t</c> code.</summary>
    public static string ct_status_string(int status)
        => Marshal.PtrToStringAnsi(ct_status_string_native(status)) ?? string.Empty;

    /// <summary>Message from the most recent failed call on this thread.</summary>
    public static string ct_get_last_error()
        => Marshal.PtrToStringAnsi(ct_get_last_error_native()) ?? string.Empty;

    /// <summary>Throw <see cref="TacudaException"/> when <paramref name="status"/> is non-zero.</summary>
    public static void ThrowOnError(int status, string function)
    {
        if (status != 0)
        {
            throw new TacudaException(function, status, ct_status_string(status), ct_get_last_error());
        }
    }
}

/// <summary>Raised when a native TACUDA call reports a non-success status.</summary>
public sealed class TacudaException : Exception
{
    public string Function { get; }

    public int Status { get; }

    public TacudaException(string function, int status, string statusText, string detail)
        : base($"{function} failed: {statusText} (status {status}){(string.IsNullOrEmpty(detail) ? "" : $": {detail}")}")
    {
        Function = function;
        Status = status;
    }
}
