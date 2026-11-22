/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_CORE_PORTABLERANDOM_H
#define TACUDA_OPT_CORE_PORTABLERANDOM_H

#include <cstddef>
#include <cstdint>
#include <random>

namespace topt {

// The engine's output sequence is standardized. The mappings used by
// std::uniform_*_distribution and std::normal_distribution are not.
class PortableRandom {
public:
  explicit PortableRandom(std::uint64_t seed) : engine_(seed) {}

  std::size_t index(std::uint64_t count) {
    // Discard the incomplete interval before reducing modulo count.
    const std::uint64_t threshold = (std::uint64_t{0} - count) % count;
    std::uint64_t bits;
    do { bits = engine_(); } while (bits < threshold);
    return static_cast<std::size_t>(bits % count);
  }

  double unit() { return static_cast<double>(engine_() >> 11) * 0x1p-53; }

  double normal() {
    // A unit exponential proposal x >= 0, accepted with probability
    // exp(-(x-1)^2/2), has density proportional to exp(-x^2/2).
    // Use a second exponential for that test and a random sign for symmetry.
    double x, offset;
    do {
      x = exponential();
      offset = x - 1.0;
    } while (2.0 * exponential() < offset * offset);
    return unit() < 0.5 ? -x : x;
  }

private:
  double exponential() {
    // Von Neumann's exponential sampler. An even-length decreasing run
    // accepts the proposed fraction u with probability exp(-u). A rejection
    // advances the integer part. All draws use exact binary fractions.
    for (unsigned whole = 0; ; ++whole) {
      const double fraction = unit();
      double previous = fraction;
      bool even = true;
      for (double next = unit(); next < previous; next = unit()) {
        previous = next;
        even = !even;
      }
      if (even) return whole + fraction;
    }
  }

  std::mt19937_64 engine_;
};

} // namespace topt

#endif // TACUDA_OPT_CORE_PORTABLERANDOM_H
