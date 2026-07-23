/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <utils/TacudaConfig.h>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <string>

namespace tacuda {

Config& Config::instance() {
    static Config cfg = []() {
        Config c;
        c.loadFromEnvironment();
        return c;
    }();
    return cfg;
}

void Config::loadFromEnvironment() {
    // Helper lambda
    auto getInt = [](const char* name, int defaultVal) -> int {
        if (const char* s = std::getenv(name)) return std::max(0, std::atoi(s));
        return defaultVal;
    };
    auto getBool = [](const char* name, bool defaultVal) -> bool {
        if (const char* s = std::getenv(name)) return s[0] != '0';
        return defaultVal;
    };
    auto getDType = [](const char* name, PrecisionDType defaultVal) -> PrecisionDType {
        const char* s = std::getenv(name);
        if (!s) return defaultVal;
        std::string v(s);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (v == "f16" || v == "fp16" || v == "half") {
            return PrecisionDType::F16;
        }
        if (v == "f64" || v == "fp64" || v == "double") {
            return PrecisionDType::F64;
        }
        return PrecisionDType::F32;
    };

    // Global
    defaultBlockSize = getInt("TACUDA_BLOCK_SIZE", defaultBlockSize);

    // Batch (generic — individual indicators fall back to these)
    batch.count     = std::max(1, getInt("TACUDA_BATCH_COUNT", batch.count));
    batch.seriesLen = getInt("TACUDA_SERIES_LEN", batch.seriesLen);
    batch.timeMajor = getBool("TACUDA_TIME_MAJOR", batch.timeMajor);

    // Thresholds
    thresholds.sequentialLimit = getInt("TACUDA_SEQ_T", thresholds.sequentialLimit);
    thresholds.warpLimit       = getInt("TACUDA_WARP_T", thresholds.warpLimit);
    thresholds.prefixSumLimit  = getInt("TACUDA_PREFIX_T", thresholds.prefixSumLimit);

    // Features
    features.preferPrefixSum    = getBool("TACUDA_PREFER_PREFIX", features.preferPrefixSum);
    features.enableWarpKernels  = getBool("TACUDA_ENABLE_WARP", features.enableWarpKernels);
    features.useFusedStochastic = getBool("TACUDA_STOCH_USE_FUSED", features.useFusedStochastic);

    // Precision
    precision.storageDType = getDType("TACUDA_STORAGE_DTYPE", precision.storageDType);
    precision.computeDType = getDType("TACUDA_COMPUTE_DTYPE", precision.computeDType);
    // Guard: storage=f16 supports compute=f32 only in current implementation.
    if (precision.storageDType == PrecisionDType::F16 && precision.computeDType != PrecisionDType::F32) {
        precision.computeDType = PrecisionDType::F32;
    }

    // Per-indicator batch overrides (backward compatible)
    auto loadBatch = [&](const char* countEnv, const char* lenEnv, int& c, int& l) {
        c = std::max(1, getInt(countEnv, batch.count));
        l = getInt(lenEnv, batch.seriesLen);
    };

    // RSI
    {
        int c = batch.count, l = batch.seriesLen;
        loadBatch("TACUDA_RSI_BATCH_COUNT", "TACUDA_RSI_SERIES_LEN", c, l);
        // Store in batch (RSI uses batch config at runtime)
    }

    // SMA
    sma.blockSize             = getInt("TACUDA_SMA_BLOCK_SIZE", sma.blockSize);
    sma.smallWindowThreshold  = getInt("TACUDA_SMA_SMALL_T", sma.smallWindowThreshold);
    sma.warpWindowThreshold   = getInt("TACUDA_SMA_WARP_T", sma.warpWindowThreshold);
    sma.gridStrideFactor      = getInt("TACUDA_SMA_GRID_FACTOR", sma.gridStrideFactor);
    sma.maxSharedBytes        = getInt("TACUDA_SMA_MAX_SHARED", sma.maxSharedBytes);
    sma.preferPrefixForLarge  = getBool("TACUDA_SMA_PREFER_PREFIX", sma.preferPrefixForLarge);
    sma.enableWarp            = getBool("TACUDA_SMA_ENABLE_WARP", sma.enableWarp);
    sma.prefixPreferThreshold = getInt("TACUDA_SMA_PREFIX_T", sma.prefixPreferThreshold);

    // WMA
    wma.blockSize             = getInt("TACUDA_WMA_BLOCK_SIZE", wma.blockSize);
    wma.smallWindowThreshold  = getInt("TACUDA_WMA_SMALL_T", wma.smallWindowThreshold);
    wma.warpWindowThreshold   = getInt("TACUDA_WMA_WARP_T", wma.warpWindowThreshold);
    wma.gridStrideFactor      = getInt("TACUDA_WMA_GRID_FACTOR", wma.gridStrideFactor);
    wma.maxSharedBytes        = getInt("TACUDA_WMA_MAX_SHARED", wma.maxSharedBytes);
    wma.preferPrefixForLarge  = getBool("TACUDA_WMA_PREFER_PREFIX", wma.preferPrefixForLarge);
    wma.enableWarp            = getBool("TACUDA_WMA_ENABLE_WARP", wma.enableWarp);
    wma.prefixPreferThreshold = getInt("TACUDA_WMA_PREFIX_T", wma.prefixPreferThreshold);
    wma.enableVector          = getBool("TACUDA_WMA_ENABLE_VEC", wma.enableVector);

    // EMA
    ema.blockSize           = getInt("TACUDA_EMA_BLOCK_SIZE", ema.blockSize);
    ema.sequentialThreshold = getInt("TACUDA_EMA_SEQ_T", ema.sequentialThreshold);
    ema.warpThreshold       = getInt("TACUDA_EMA_WARP_T", ema.warpThreshold);
    ema.algo                = getInt("TACUDA_EMA_ALGO", ema.algo);

    // KAMA
    kama.sequentialThreshold = getInt("TACUDA_KAMA_SEQ_T", kama.sequentialThreshold);
    kama.warpThreshold       = getInt("TACUDA_KAMA_WARP_T", kama.warpThreshold);

    // T3
    t3.blockSize           = getInt("TACUDA_T3_BLOCK_SIZE", t3.blockSize);
    t3.sequentialThreshold = getInt("TACUDA_T3_SEQ_T", t3.sequentialThreshold);
    t3.algo                = getInt("TACUDA_T3_ALGO", t3.algo);

    // MAMA
    mama.algo         = getInt("TACUDA_MAMA_ALGO", mama.algo);
    mama.batchCount   = std::max(1, getInt("TACUDA_MAMA_BATCH_COUNT", mama.batchCount));
    mama.seriesLength = getInt("TACUDA_MAMA_SERIES_LEN", mama.seriesLength);
}

} // namespace tacuda
