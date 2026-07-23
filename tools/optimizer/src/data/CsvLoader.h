/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_DATA_CSVLOADER_H
#define TACUDA_OPT_DATA_CSVLOADER_H

#include <string>

#include "core/Types.h"

namespace topt {

// Loads an OHLCV series from a CSV file.
//
// Accepted layouts (comma or semicolon separated):
//   - with a header: columns matched case-insensitively by name
//     (timestamp/date/time/datetime, open, high, low, close, volume/vol);
//   - without a header: [timestamp,]open,high,low,close[,volume].
// Timestamps may be unix seconds, unix milliseconds or
// "YYYY-MM-DD[ HH:MM[:SS]]" / "YYYY-MM-DDTHH:MM[:SS]".
// Rows with high < low or non-finite prices are rejected.
// Throws std::runtime_error with file/line context on errors.
Series load_csv(const std::string& path);

} // namespace topt

#endif // TACUDA_OPT_DATA_CSVLOADER_H
