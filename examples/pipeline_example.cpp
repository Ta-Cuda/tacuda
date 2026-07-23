/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * Pipeline Example - Demonstrates using tacuda::Context for efficient
 * multi-indicator computation without intermediate host transfers.
 * 
 * This example shows how to:
 * 1. Upload data to GPU once
 * 2. Compute multiple indicators in a chain
 * 3. Download only the final results
 */

#include <tacuda/Context.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

using namespace tacuda;

// Generate synthetic OHLCV data
void generateOHLCV(std::vector<float>& open, std::vector<float>& high,
                   std::vector<float>& low, std::vector<float>& close,
                   std::vector<float>& volume, size_t size) {
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::uniform_real_distribution<float> vol_dist(100000.0f, 2000000.0f);
    
    float price = 100.0f;
    
    for (size_t i = 0; i < size; ++i) {
        float change = noise(rng);
        price = std::max(1.0f, price + change);
        
        float o = price;
        float h = price + std::abs(noise(rng)) * 2;
        float l = price - std::abs(noise(rng)) * 2;
        float c = price + noise(rng);
        
        // Ensure OHLC consistency
        h = std::max({o, h, c});
        l = std::min({o, l, c});
        
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c;
        volume[i] = vol_dist(rng);
        
        price = c;
    }
}

int main() {
    const size_t SIZE = 1000000;  // 1 million bars
    
    std::cout << "=== tacuda Pipeline Example ===" << std::endl;
    std::cout << "Data size: " << SIZE << " bars" << std::endl << std::endl;
    
    // Generate test data
    std::vector<float> h_open(SIZE), h_high(SIZE), h_low(SIZE), h_close(SIZE), h_volume(SIZE);
    generateOHLCV(h_open, h_high, h_low, h_close, h_volume, SIZE);
    
    // Output buffers
    std::vector<float> h_rsi(SIZE), h_macd(SIZE), h_signal(SIZE), h_atr(SIZE);
    std::vector<float> h_ema_fast(SIZE), h_ema_slow(SIZE), h_trend_signal(SIZE);
    
    // ========================================================================
    // Method 1: Traditional approach (multiple host transfers)
    // ========================================================================
    std::cout << "Method 1: Traditional (separate calls with host transfers)" << std::endl;
    
    auto start1 = std::chrono::high_resolution_clock::now();
    {
        // Each indicator call does: host->device, compute, device->host
        // This is inefficient for multiple indicators
        
        // ... (would require C API calls here)
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    
    // ========================================================================
    // Method 2: Pipeline approach (single upload, multiple compute, single download)
    // ========================================================================
    std::cout << "Method 2: Pipeline (data stays on GPU)" << std::endl;
    
    auto start2 = std::chrono::high_resolution_clock::now();
    {
        // Create context
        Context ctx;
        
        // Upload OHLCV data to GPU (ONE transfer)
        auto ohlcv = ctx.uploadOHLCV(h_open.data(), h_high.data(), h_low.data(),
                                      h_close.data(), h_volume.data(), SIZE);
        
        // Compute multiple indicators (ALL on GPU, no transfers)
        auto rsi = ctx.rsi(ohlcv.close, 14);
        auto atr = ctx.atr(ohlcv.high, ohlcv.low, ohlcv.close, 14);
        
        // MACD with components
        auto macd_result = ctx.macd(ohlcv.close, 12, 26, 9);
        
        // Moving average crossover signal
        auto ema_fast = ctx.ema(ohlcv.close, 12);
        auto ema_slow = ctx.ema(ohlcv.close, 26);
        auto trend_signal = ctx.subtract(ema_fast, ema_slow);
        
        // Stochastic
        auto stoch_result = ctx.stoch(ohlcv.high, ohlcv.low, ohlcv.close, 14, 3, 3);
        
        // Bollinger Bands
        auto bb = ctx.bbands(ohlcv.close, 20, 2.0, 2.0, 0);
        
        // ADX
        auto adx = ctx.adx(ohlcv.high, ohlcv.low, ohlcv.close, 14);
        
        // Volume indicators
        auto obv = ctx.obv(ohlcv.close, ohlcv.volume);
        auto mfi = ctx.mfi(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, 14);
        
        // Download only what we need (selective transfers)
        ctx.download(rsi, h_rsi.data());
        ctx.download(atr, h_atr.data());
        ctx.download(macd_result.macd, h_macd.data());
        ctx.download(macd_result.signal, h_signal.data());
        ctx.download(trend_signal, h_trend_signal.data());
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "Pipeline time: " << duration2.count() << " ms" << std::endl;
    
    // ========================================================================
    // Verify results
    // ========================================================================
    std::cout << std::endl << "Sample results (last 5 values):" << std::endl;
    std::cout << "Index\tRSI\t\tATR\t\tMACD\t\tTrend Signal" << std::endl;
    for (size_t i = SIZE - 5; i < SIZE; ++i) {
        std::cout << i << "\t" 
                  << h_rsi[i] << "\t" 
                  << h_atr[i] << "\t"
                  << h_macd[i] << "\t"
                  << h_trend_signal[i] << std::endl;
    }
    
    // ========================================================================
    // Advanced: Chained computations
    // ========================================================================
    std::cout << std::endl << "=== Advanced: Chained Computations ===" << std::endl;
    {
        Context ctx;
        
        auto close = ctx.upload(h_close.data(), SIZE);
        
        // Chain: Close -> RSI -> EMA of RSI -> Signal
        auto rsi = ctx.rsi(close, 14);
        auto rsi_ema = ctx.ema(rsi, 9);
        auto rsi_signal = ctx.subtract(rsi, rsi_ema);
        
        // Chain: Close -> ATR -> Normalized by Close
        auto high = ctx.upload(h_high.data(), SIZE);
        auto low = ctx.upload(h_low.data(), SIZE);
        auto atr = ctx.atr(high, low, close, 14);
        auto atr_pct = ctx.divide(atr, close);  // ATR as % of price
        auto atr_pct_scaled = ctx.scale(atr_pct, 100.0f);  // As percentage
        
        // Download
        std::vector<float> h_rsi_signal(SIZE), h_atr_pct(SIZE);
        ctx.download(rsi_signal, h_rsi_signal.data());
        ctx.download(atr_pct_scaled, h_atr_pct.data());
        
        std::cout << "RSI Signal (last 3): ";
        for (size_t i = SIZE - 3; i < SIZE; ++i) {
            std::cout << h_rsi_signal[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "ATR % (last 3): ";
        for (size_t i = SIZE - 3; i < SIZE; ++i) {
            std::cout << h_atr_pct[i] << "% ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl << "=== Pipeline Example Complete ===" << std::endl;
    
    return 0;
}

