/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_CONFIG_H
#define TACUDA_CONFIG_H

/**
 * @file TacudaConfig.h
 * @brief Unified configuration for the TaCuda library.
 *
 * Replaces dozens of per-indicator environment variables with a single
 * configuration structure. Still supports backward-compatible loading
 * from environment variables via loadFromEnvironment().
 *
 * Usage:
 *   // Programmatic:
 *   auto& cfg = tacuda::Config::instance();
 *   cfg.batch.count = 4;
 *   cfg.batch.seriesLen = 1024;
 *
 *   // Or load from environment (called once at first access):
 *   // TACUDA_BATCH_COUNT=4 TACUDA_SERIES_LEN=1024 ./myapp
 */

#include <cstddef>

namespace tacuda {

enum class PrecisionDType {
    F16 = 0,
    F32 = 1,
    F64 = 2,
};

struct Config {
    // ========================================================================
    // Global defaults
    // ========================================================================
    int defaultBlockSize = 256;

    // ========================================================================
    // Batch processing (shared by RSI, MACD, ATR, NATR, BBANDS, Stochastic, MAMA)
    // ========================================================================
    struct Batch {
        int count = 1;
        int seriesLen = 0;      // 0 = auto (size / count)
        bool timeMajor = false;
    } batch;

    // ========================================================================
    // Algorithm selection thresholds
    // ========================================================================
    struct Thresholds {
        int sequentialLimit = 10000;    // Below this: may use simpler path
        int warpLimit = 100000;         // Warp-optimized kernel threshold
        int prefixSumLimit = 96;        // SMA prefix-sum threshold
    } thresholds;

    // ========================================================================
    // Feature flags
    // ========================================================================
    struct Features {
        bool preferPrefixSum = true;
        bool enableWarpKernels = true;
        bool useFusedStochastic = true;
    } features;

    // ========================================================================
    // Precision policy (runtime selection)
    // ========================================================================
    struct Precision {
        PrecisionDType storageDType = PrecisionDType::F32;
        PrecisionDType computeDType = PrecisionDType::F32;
    } precision;

    // ========================================================================
    // Pool settings
    // ========================================================================
    struct Pool {
        size_t maxCacheBytes = static_cast<size_t>(-1); // SIZE_MAX
    } pool;

    // ========================================================================
    // SMA-specific (advanced tuning)
    // ========================================================================
    struct SMA {
        int blockSize = 256;
        int smallWindowThreshold = 32;
        int warpWindowThreshold = 96;
        int gridStrideFactor = 4;
        int maxSharedBytes = 0;
        bool preferPrefixForLarge = true;
        bool enableWarp = true;
        int prefixPreferThreshold = 96;
    } sma;

    // ========================================================================
    // WMA-specific (advanced tuning)
    // ========================================================================
    struct WMA {
        int blockSize = 256;
        int smallWindowThreshold = 32;
        int warpWindowThreshold = 96;
        int gridStrideFactor = 4;
        int maxSharedBytes = 0;
        bool preferPrefixForLarge = true;
        bool enableWarp = true;
        int prefixPreferThreshold = 96;
        bool enableVector = false;
    } wma;

    // ========================================================================
    // EMA-specific
    // ========================================================================
    struct EMA {
        int blockSize = 256;
        int sequentialThreshold = 10000;
        int warpThreshold = 100000;
        int algo = 0;   // 0=Auto, 3=ParallelScan (always used now)
    } ema;

    // ========================================================================
    // KAMA-specific
    // ========================================================================
    struct KAMA {
        int sequentialThreshold = 2048;
        int warpThreshold = 4096;
    } kama;

    // ========================================================================
    // T3-specific
    // ========================================================================
    struct T3 {
        int blockSize = 256;
        int sequentialThreshold = 50000;
        int algo = 0;   // 0=Auto, 2=EmaHybrid (always used now)
    } t3;

    // ========================================================================
    // MAMA-specific
    // ========================================================================
    struct MAMA {
        int algo = 0;            // 0=Sequential, 1=Batched
        int batchCount = 1;
        int seriesLength = 0;
    } mama;

    // ========================================================================
    // Singleton access
    // ========================================================================
    static Config& instance();

    // ========================================================================
    // Load from environment (backward compatible)
    // ========================================================================
    void loadFromEnvironment();
};

} // namespace tacuda

#endif // TACUDA_CONFIG_H
