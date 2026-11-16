/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Build-time host-libm coefficients. A zero-amplitude DC phase depends on
// rounding of these weights, so libdevice approximations are not interchangeable
// with the host math used by TA-Lib. No runtime host work or transfers are needed.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  std::ofstream output(argv[1]);
  if (!output) return 1;
  output << "// Generated with the build host math library. Do not edit.\n"
            "#include <utils/HilbertPhaseTable.cuh>\n"
            "namespace tacuda { namespace hilbert {\n";
  // Volatile function pointers prevent combining sin/cos or folding them with
  // a different compile-time math implementation.
  double (*volatile sine)(double) = static_cast<double (*)(double)>(&std::sin);
  double (*volatile cosine)(double) = static_cast<double (*)(double)>(&std::cos);
  for (bool isSine : {true, false}) {
    output << "__device__ __constant__ const double "
           << (isSine ? "phaseSin" : "phaseCos") << "[1275] = {\n";
    for (int period = 1; period <= 50; ++period) {
      for (int i = 0; i < period; ++i) {
        const double phase = (static_cast<double>(i) * 6.283185307179586) / period;
        const double value = isSine ? sine(phase) : cosine(phase);
        output << std::hexfloat << value << ",\n";
      }
    }
    output << "};\n";
  }
  output << "}}\n";
  output.flush();
  return output ? 0 : 1;
}
