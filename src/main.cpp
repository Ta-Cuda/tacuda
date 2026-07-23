/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// example_app: minimal smoke example built on the PUBLIC C API only.
// (Indicator classes like tacuda::SMA are internal — since the DLL exports
// just the C ABI + Context, linking against them from an executable is not
// supported.)
#include <cmath>
#include <iostream>
#include <vector>

#include <tacuda.h>
#include <tacuda_device.h>

namespace {
void check(ctStatus_t st, const char* what) {
    if (st != CT_STATUS_SUCCESS) {
        std::cerr << what << " failed: " << ct_status_string(st) << " ("
                  << ct_get_last_error() << ")\n";
        std::exit(1);
    }
}

void print_head(const char* name, const std::vector<float>& v) {
    std::cout << name << "[0..4]: ";
    for (int i = 0; i < 5; ++i) std::cout << v[static_cast<size_t>(i)] << (i < 4 ? ", " : "\n");
}
}  // namespace

int main() {
    std::cout << "tacuda " << ct_version() << "\n";

    const int N = 1024;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[static_cast<size_t>(i)] = std::sin(0.01f * i) + 0.5f * std::cos(0.03f * i);

    std::vector<float> out(N, 0.0f);

    // Host API: one call per indicator.
    check(ct_sma(in.data(), out.data(), N, 14), "ct_sma");
    print_head("SMA", out);

    check(ct_momentum(in.data(), out.data(), N, 10), "ct_momentum");
    print_head("Momentum", out);

    check(ct_macd_line(in.data(), out.data(), N, 12, 26), "ct_macd_line");
    print_head("MACD line", out);

    // Device API: zero-copy pipeline on GPU buffers.
    float* dIn = ct_device_alloc(N);
    float* dOut = ct_device_alloc(N);
    if (!dIn || !dOut) {
        std::cerr << "ct_device_alloc failed\n";
        return 1;
    }
    check(ct_device_upload(dIn, in.data(), N, nullptr), "ct_device_upload");
    check(ct_rsi_d(dIn, dOut, N, 14, nullptr), "ct_rsi_d");
    check(ct_device_download(out.data(), dOut, N, nullptr), "ct_device_download");
    check(ct_device_sync(nullptr), "ct_device_sync");
    print_head("RSI (device)", out);

    ct_device_free(dIn);
    ct_device_free(dOut);
    ct_device_buffer_pool_cleanup();
    return 0;
}
