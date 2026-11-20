/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Build-time host-libm coefficients. A zero-amplitude DC phase depends on
// rounding of these weights, so libdevice approximations are not interchangeable
// with the host math used by TA-Lib. No runtime host work or transfers are needed.
// The device constants and the host copy are written from the same text, so the
// GPU kernels and the host API read identical weights; the host API no longer
// calls sin/cos at runtime, where the library may pick an FMA variant.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

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
    std::ostringstream values;
    values << std::hexfloat;
    for (int period = 1; period <= 50; ++period) {
      for (int i = 0; i < period; ++i) {
        const double phase = (static_cast<double>(i) * 6.283185307179586) / period;
        values << (isSine ? sine(phase) : cosine(phase)) << ",\n";
      }
    }
    const std::string name = isSine ? "phaseSin" : "phaseCos";
    output << "__device__ __constant__ const double " << name << "[1275] = {\n"
           << values.str() << "};\n"
           << "const double " << name << "Host[1275] = {\n"
           << values.str() << "};\n";
  }
  output << "}}\n";
  output.flush();
  return output ? 0 : 1;
}
