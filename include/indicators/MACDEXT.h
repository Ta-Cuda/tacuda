/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef MACDEXT_H
#define MACDEXT_H

#include "Indicator.h"
#include "MA.h"

namespace tacuda {
class MACDEXT : public Indicator {
public:
    MACDEXT(int fastPeriod, int slowPeriod, int signalPeriod, MAType type);
    /**
     * @brief TA-Lib MACDEXT: independent MA type per line. Like TA-Lib, the
     *        fast/slow pairs are swapped when fastPeriod > slowPeriod.
     */
    MACDEXT(int fastPeriod, MAType fastType, int slowPeriod, MAType slowType,
            int signalPeriod, MAType signalType);
    void calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) override;
    void calculate(const float* input, float* macd, float* signal, float* hist,
                   int size, cudaStream_t stream = 0) noexcept(false);

    /** @brief TA-Lib lookback (first valid index) of a moving-average type. */
    static int maTypeLookback(MAType type, int period);

private:
    int fastPeriod;
    int slowPeriod;
    int signalPeriod;
    MAType fastType;
    MAType slowType;
    MAType signalType;
};

} // namespace tacuda

#endif
