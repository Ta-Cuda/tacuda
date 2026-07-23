/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <limits>

#include "tacuda.h"
#include "tacuda_device.h"
#include <utils/ErrorState.h>
#include <utils/Precision.h>
#include <indicators/AD.h>
#include <indicators/ACCBANDS.h>
#include <indicators/ADOSC.h>
#include <indicators/ADX.h>
#include <indicators/ADXR.h>
#include <indicators/AVGDEV.h>
#include <indicators/APO.h>
#include <indicators/ATR.h>
#include <indicators/AbandonedBaby.h>
#include <indicators/AdvanceBlock.h>
#include <indicators/Aroon.h>
#include <indicators/AroonOscillator.h>
#include <indicators/AvgPrice.h>
#include <indicators/BBANDS.h>
#include <indicators/BOP.h>
#include <indicators/BearishEngulfing.h>
#include <indicators/BeltHold.h>
#include <indicators/Beta.h>
#include <indicators/Breakaway.h>
#include <indicators/BullishEngulfing.h>
#include <indicators/CCI.h>
#include <indicators/CMO.h>
#include <indicators/Change.h>
#include <indicators/ClosingMarubozu.h>
#include <indicators/ConcealBabySwallow.h>
#include <indicators/Correl.h>
#include <indicators/CounterAttack.h>
#include <indicators/CDL3OUTSIDE.h>
#include <indicators/DEMA.h>
#include <indicators/DX.h>
#include <indicators/DarkCloudCover.h>
#include <indicators/Doji.h>
#include <indicators/IMI.h>
#include <indicators/DojiStar.h>
#include <indicators/DragonflyDoji.h>
#include <indicators/EMA.h>
#include <indicators/Engulfing.h>
#include <indicators/EveningDojiStar.h>
#include <indicators/EveningStar.h>
#include <indicators/GapSideSideWhite.h>
#include <indicators/GravestoneDoji.h>
#include <indicators/HT_DCPERIOD.h>
#include <indicators/HT_DCPHASE.h>
#include <indicators/HT_PHASOR.h>
#include <indicators/HT_SINE.h>
#include <indicators/HT_TRENDLINE.h>
#include <indicators/HT_TRENDMODE.h>
#include <utils/HilbertCPU.h>
#include <indicators/Hammer.h>
#include <indicators/HangingMan.h>
#include <indicators/Harami.h>
#include <indicators/HaramiCross.h>
#include <indicators/HighWave.h>
#include <indicators/Hikkake.h>
#include <indicators/HikkakeMod.h>
#include <indicators/HomingPigeon.h>
#include <indicators/IdenticalThreeCrows.h>
#include <indicators/InNeck.h>
#include <indicators/InvertedHammer.h>
#include <indicators/KAMA.h>
#include <indicators/Kicking.h>
#include <indicators/KickingByLength.h>
#include <indicators/LINEARREG.h>
#include <indicators/LINEARREG_ANGLE.h>
#include <indicators/LINEARREG_INTERCEPT.h>
#include <indicators/LINEARREG_SLOPE.h>
#include <indicators/LadderBottom.h>
#include <indicators/LongLeggedDoji.h>
#include <indicators/LongLine.h>
#include <indicators/MA.h>
#include <indicators/MAVP.h>
#include <indicators/MACD.h>
#include <indicators/MACDEXT.h>
#include <indicators/MACDFIX.h>
#include <indicators/MAMA.h>
#include <indicators/MAX.h>
#include <indicators/MAXINDEX.h>
#include <indicators/MFI.h>
#include <indicators/MIN.h>
#include <indicators/MININDEX.h>
#include <indicators/MINMAX.h>
#include <indicators/MINMAXINDEX.h>
#include <indicators/Marubozu.h>
#include <indicators/MatHold.h>
#include <indicators/MatchingLow.h>
#include <indicators/MedPrice.h>
#include <indicators/MIDPOINT.h>
#include <indicators/MIDPRICE.h>
#include <indicators/MinusDI.h>
#include <indicators/MinusDM.h>
#include <indicators/Momentum.h>
#include <indicators/NVI.h>
#include <indicators/MorningDojiStar.h>
#include <indicators/MorningStar.h>
#include <indicators/NATR.h>
#include <indicators/PVI.h>
#include <indicators/OBV.h>
#include <indicators/OnNeck.h>
#include <indicators/PPO.h>
#include <indicators/PVO.h>
#include <indicators/Piercing.h>
#include <indicators/PlusDI.h>
#include <indicators/PlusDM.h>
#include <indicators/ROC.h>
#include <indicators/ROCP.h>
#include <indicators/ROCR.h>
#include <indicators/ROCR100.h>
#include <indicators/RSI.h>
#include <indicators/RickshawMan.h>
#include <indicators/RiseFall3Methods.h>
#include <indicators/SAR.h>
#include <indicators/SAREXT.h>
#include <indicators/SMA.h>
#include <indicators/SUM.h>
#include <indicators/SeparatingLines.h>
#include <indicators/ShootingStar.h>
#include <indicators/ShortLine.h>
#include <indicators/SpinningTop.h>
#include <indicators/StalledPattern.h>
#include <indicators/StdDev.h>
#include <indicators/StickSandwich.h>
#include <indicators/StochRSI.h>
#include <indicators/Stochastic.h>
#include <indicators/StochasticFast.h>
#include <indicators/T3.h>
#include <indicators/TEMA.h>
#include <indicators/TRANGE.h>
#include <indicators/TRIMA.h>
#include <indicators/TRIX.h>
#include <indicators/TSF.h>
#include <indicators/Takuri.h>
#include <indicators/TasukiGap.h>
#include <indicators/ThreeBlackCrows.h>
#include <indicators/ThreeInside.h>
#include <indicators/ThreeLineStrike.h>
#include <indicators/ThreeStarsInSouth.h>
#include <indicators/ThreeWhiteSoldiers.h>
#include <indicators/Thrusting.h>
#include <indicators/Tristar.h>
#include <indicators/TwoCrows.h>
#include <indicators/TypPrice.h>
#include <indicators/ULTOSC.h>
#include <indicators/Unique3River.h>
#include <indicators/UpsideGap2Crows.h>
#include <indicators/VAR.h>
#include <indicators/WMA.h>
#include <indicators/WclPrice.h>
#include <indicators/WILLR.h>
#include <indicators/XSideGap3Methods.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/SarTaLib.h>

struct PooledDeleter {
  // Stream-aware release: the buffer re-enters the pool only after work
  // already enqueued on this stream completes, so early-error returns with
  // async copies/kernels still in flight cannot cause reuse-while-busy.
  cudaStream_t stream = 0;
  void operator()(float *ptr) const noexcept {
    if (ptr) DeviceBufferPool::instance().releaseAsync(ptr, stream);
  }
};
using DeviceBuffer = std::unique_ptr<float, PooledDeleter>;

static bool acquire_buffer(DeviceBuffer &buffer, size_t elements) {
  float *tmp = nullptr;
  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(elements * sizeof(float)));
  } catch (...) {
    return false;
  }
  buffer.reset(tmp);
  return true;
}

// Helper: classify a caught exception into the appropriate status code
// and store the error message in thread-local storage.
static ctStatus_t classify_exception(const char* context = nullptr) {
  try { throw; } // re-throw current exception
  catch (const std::invalid_argument& e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_INVALID_ARGUMENT;
  }
  catch (const std::runtime_error& e) {
    // CudaUtils.h throws runtime_error for CUDA errors
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_DEVICE_ERROR;
  }
  catch (const std::exception& e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_KERNEL_FAILED;
  }
  catch (...) {
    tacuda::detail::set_last_error(context ? context : "unknown error");
    return CT_STATUS_KERNEL_FAILED;
  }
}

// Every extern "C" entry point funnels through this guard: the thread-local
// error is cleared, and no C++ exception (indicator constructors included)
// may ever cross the C ABI — everything is translated into a ct_status.
template <typename Fn>
static ctStatus_t api_guard(const char* what, Fn&& fn) noexcept {
  tacuda::detail::clear_last_error();
  try {
    return fn();
  } catch (...) {
    return classify_exception(what);
  }
}

static ctStatus_t run_indicator(tacuda::Indicator &ind, const float *h_in, float *h_out,
                                int size, int outMultiple = 1,
                                cudaStream_t stream = 0) {
  tacuda::detail::clear_last_error();
  if (!h_in || !h_out || size <= 0 || outMultiple <= 0) {
    tacuda::detail::set_last_error("null pointer or non-positive size");
    return CT_STATUS_INVALID_ARGUMENT;
  }
  const size_t inBytes = static_cast<size_t>(size) * sizeof(float);
  const size_t outBytes =
      static_cast<size_t>(size) * static_cast<size_t>(outMultiple) * sizeof(float);
  DeviceBuffer d_in{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(inBytes));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device input buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_in.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(outBytes));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device output buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_in.get(), h_in, inBytes,
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_in.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception("indicator calculate failed");
  }

  err = cudaMemcpyAsync(h_out, d_out.get(), outBytes,
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_DEVICE_ERROR;
  }

  return CT_STATUS_SUCCESS;
}

static ctStatus_t run_unary_ex_host(
    ctStatus_t (*fn)(const void*, void*, int, int, ctDataType_t, ctDataType_t, cudaStream_t),
    const void* host_input,
    void* host_output,
    int size,
    int period,
    ctDataType_t storageType,
    ctDataType_t computeType,
    cudaStream_t stream) {
  if (!host_input || !host_output || size <= 0) return CT_STATUS_INVALID_ARGUMENT;
  if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
    return CT_STATUS_UNSUPPORTED;
  }

  void* d_in = ct_device_alloc_ex(size, storageType);
  void* d_out = ct_device_alloc_ex(size, storageType);
  if (!d_in || !d_out) {
    if (d_in) ct_device_free_ex(d_in);
    if (d_out) ct_device_free_ex(d_out);
    return CT_STATUS_ALLOC_FAILED;
  }

  ctStatus_t st = ct_device_upload_ex(d_in, host_input, size, storageType, stream);
  if (st != CT_STATUS_SUCCESS) {
    ct_device_free_ex(d_in);
    ct_device_free_ex(d_out);
    return st;
  }

  st = fn(d_in, d_out, size, period, storageType, computeType, stream);
  if (st != CT_STATUS_SUCCESS) {
    ct_device_free_ex(d_in);
    ct_device_free_ex(d_out);
    return st;
  }

  st = ct_device_download_ex(host_output, d_out, size, storageType, stream);
  if (st != CT_STATUS_SUCCESS) {
    ct_device_free_ex(d_in);
    ct_device_free_ex(d_out);
    return st;
  }
  cudaError_t err = cudaStreamSynchronize(stream);
  ct_device_free_ex(d_in);
  ct_device_free_ex(d_out);
  return (err == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_DEVICE_ERROR;
}

template <typename Indicator>
static ctStatus_t run_indicator_device(Indicator &ind, const float *d_in,
                                       float *d_out, int size,
                                       cudaStream_t stream = 0) {
  tacuda::detail::clear_last_error();
  try {
    ind.calculate(d_in, d_out, size, stream);
  } catch (...) {
    return classify_exception("indicator calculate (device) failed");
  }

  return CT_STATUS_SUCCESS;
}

template <typename T>
static ctStatus_t run_ohlc_indicator(T &ind, const float *h_open,
                                     const float *h_high, const float *h_low,
                                     const float *h_close, float *h_out,
                                     int size, int outMultiple = 1,
                                     cudaStream_t stream = 0) {
  tacuda::detail::clear_last_error();
  DeviceBuffer d_open{nullptr, PooledDeleter{stream}}, d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}},
      d_close{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device open buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_open.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device high buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_high.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device low buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_low.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device close buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_close.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(
        size * outMultiple * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device output buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_open.get(), h_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_high.get(), h_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), h_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), h_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(),
                  d_out.get(), size, stream);
  } catch (...) {
    return classify_exception("OHLC indicator calculate failed");
  }

  err = cudaMemcpyAsync(h_out, d_out.get(),
                        size * outMultiple * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_DEVICE_ERROR;
  }

  return CT_STATUS_SUCCESS;
}

template <typename T>
static ctStatus_t run_ohlc_indicator(T &ind, const float *h_high,
                                     const float *h_low, const float *h_close,
                                     float *h_out, int size,
                                     int outMultiple = 1,
                                     cudaStream_t stream = 0) {
  tacuda::detail::clear_last_error();
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_out{nullptr, PooledDeleter{stream}};
  float *tmp = nullptr;

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device high buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_high.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device low buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_low.reset(tmp);

  try {
    tmp = static_cast<float *>(
        DeviceBufferPool::instance().acquire(size * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device close buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_close.reset(tmp);

  try {
    tmp = static_cast<float *>(DeviceBufferPool::instance().acquire(
        size * outMultiple * sizeof(float)));
  } catch (...) {
    tacuda::detail::set_last_error("failed to allocate device output buffer");
    return CT_STATUS_ALLOC_FAILED;
  }
  d_out.reset(tmp);

  cudaError_t err = cudaMemcpyAsync(d_high.get(), h_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), h_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), h_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                  stream);
  } catch (...) {
    return classify_exception("HLC indicator calculate failed");
  }

  err = cudaMemcpyAsync(h_out, d_out.get(),
                        size * outMultiple * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_DEVICE_ERROR;
  }

  return CT_STATUS_SUCCESS;
}

extern "C" {

ctStatus_t ct_sma(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_sma", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_sma: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sma: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_sma_ex(host_input, host_output, size, period, CT_DTYPE_F32,
                   CT_DTYPE_F32, stream);

  });
}

ctStatus_t ct_sma_ex(const void *host_input, void *host_output, int size,
                     int period, ctDataType_t storageType,
                     ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_sma_ex", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_sma_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sma_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return run_unary_ex_host(&ct_sma_ex_d, host_input, host_output, size, period,
                           storageType, computeType, stream);

  });
}

ctStatus_t ct_ma(const float *host_input, float *host_output, int size,
                 int period, ctMaType_t type, cudaStream_t stream) {
  return api_guard("ct_ma", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ma: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ma: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MA ma(period, static_cast<tacuda::MAType>(type));
  return run_indicator(ma, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_mavp(const float *host_input, const float *host_periods,
                   float *host_output, int size, int minPeriod, int maxPeriod,
                   ctMaType_t type, cudaStream_t stream) {
  return api_guard("ct_mavp", [&]() -> ctStatus_t {
    if (!host_input || !host_periods || !host_output) {
      tacuda::detail::set_last_error("ct_mavp: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_mavp: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAVP mavp(minPeriod, maxPeriod, static_cast<tacuda::MAType>(type));

  DeviceBuffer d_values{nullptr, PooledDeleter{stream}}, d_periods{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  if (!acquire_buffer(d_values, size) || !acquire_buffer(d_periods, size) ||
      !acquire_buffer(d_out, size)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err =
      cudaMemcpyAsync(d_values.get(), host_input, size * sizeof(float),
                      cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaMemcpyAsync(d_periods.get(), host_periods, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mavp.calculate(d_values.get(), d_periods.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    tacuda::detail::set_last_error(cudaGetErrorString(err));
    return CT_STATUS_DEVICE_ERROR;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_wma(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_wma", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_wma: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_wma: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::WMA wma(period);
  return run_indicator(wma, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_momentum(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_momentum", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_momentum: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_momentum: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Momentum mom(period);
  return run_indicator(mom, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_change(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  return api_guard("ct_change", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_change: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_change: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Change ch(period);
  return run_indicator(ch, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_roc(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_roc", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_roc: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_roc: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROC roc(period);
  return run_indicator(roc, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_rocp(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_rocp", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_rocp: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocp: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCP rocp(period);
  return run_indicator(rocp, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_rocr(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_rocr", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_rocr: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocr: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCR rocr(period);
  return run_indicator(rocr, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_rocr100(const float *host_input, float *host_output, int size,
                      int period, cudaStream_t stream) {
  return api_guard("ct_rocr100", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_rocr100: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocr100: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCR100 rocr100(period);
  return run_indicator(rocr100, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_ema(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_ema", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ema: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ema: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_ema_ex(host_input, host_output, size, period, CT_DTYPE_F32,
                   CT_DTYPE_F32, stream);

  });
}

ctStatus_t ct_ema_ex(const void *host_input, void *host_output, int size,
                     int period, ctDataType_t storageType,
                     ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_ema_ex", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ema_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ema_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return run_unary_ex_host(&ct_ema_ex_d, host_input, host_output, size, period,
                           storageType, computeType, stream);

  });
}

ctStatus_t ct_dema(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_dema", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_dema: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_dema: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DEMA dema(period);
  return run_indicator(dema, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_tema(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_tema", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_tema: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_tema: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TEMA tema(period);
  return run_indicator(tema, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_t3(const float *host_input, float *host_output, int size,
                 int period, float vFactor, cudaStream_t stream) {
  return api_guard("ct_t3", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_t3: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_t3: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::T3 t3(period, vFactor);
  return run_indicator(t3, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_trima(const float *host_input, float *host_output, int size,
                    int period, cudaStream_t stream) {
  return api_guard("ct_trima", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_trima: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_trima: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TRIMA trima(period);
  return run_indicator(trima, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_trix(const float *host_input, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_trix", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_trix: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_trix: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TRIX trix(period);
  return run_indicator(trix, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_max(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_max", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_max: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_max: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAX mx(period);
  return run_indicator(mx, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_min(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_min", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_min: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_min: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MIN mn(period);
  return run_indicator(mn, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_maxindex(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_maxindex", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_maxindex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_maxindex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAXINDEX mi(period);
  return run_indicator(mi, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_minindex(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_minindex", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_minindex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minindex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MININDEX mi(period);
  return run_indicator(mi, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_minmax(const float *host_input, float *host_min, float *host_max,
                     int size, int period, cudaStream_t stream) {
  return api_guard("ct_minmax", [&]() -> ctStatus_t {
    if (!host_input || !host_min || !host_max) {
      tacuda::detail::set_last_error("ct_minmax: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minmax: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MINMAX mm(period);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(mm, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_min, tmp.data(), size * sizeof(float));
  std::memcpy(host_max, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_minmaxindex(const float *host_input, float *host_minidx,
                          float *host_maxidx, int size, int period, cudaStream_t stream) {
  return api_guard("ct_minmaxindex", [&]() -> ctStatus_t {
    if (!host_input || !host_minidx || !host_maxidx) {
      tacuda::detail::set_last_error("ct_minmaxindex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minmaxindex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MINMAXINDEX mm(period);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(mm, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_minidx, tmp.data(), size * sizeof(float));
  std::memcpy(host_maxidx, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_stddev(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  return api_guard("ct_stddev", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_stddev: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_stddev: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StdDev sd(period);
  return run_indicator(sd, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_avgdev(const float *host_input, float *host_output, int size,
                     int period, cudaStream_t stream) {
  return api_guard("ct_avgdev", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_avgdev: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_avgdev: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AVGDEV avg(period);
  return run_indicator(avg, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_var(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_var", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_var: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_var: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::VAR vr(period);
  return run_indicator(vr, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_sum(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_sum", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_sum: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sum: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::SUM sum(period);
  return run_indicator(sum, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_rsi(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_rsi", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_rsi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rsi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_rsi_ex(host_input, host_output, size, period, CT_DTYPE_F32,
                   CT_DTYPE_F32, stream);

  });
}

ctStatus_t ct_rsi_ex(const void *host_input, void *host_output, int size,
                     int period, ctDataType_t storageType,
                     ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_rsi_ex", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_rsi_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rsi_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return run_unary_ex_host(&ct_rsi_ex_d, host_input, host_output, size, period,
                           storageType, computeType, stream);

  });
}

ctStatus_t ct_kama(const float *host_input, float *host_output, int size,
                   int period, int fastPeriod, int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_kama", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_kama: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_kama: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::KAMA kama(period, fastPeriod, slowPeriod);
  return run_indicator(kama, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_macd_line(const float *host_input, float *host_output, int size,
                        int fastPeriod, int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_macd_line", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_macd_line: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macd_line: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (fastPeriod <= 0 || slowPeriod <= 0 || fastPeriod >= slowPeriod) {
    tacuda::detail::set_last_error("MACD: invalid fast/slow period");
    return CT_STATUS_INVALID_ARGUMENT;
  }
  if (size <= 0) {
    return CT_STATUS_SUCCESS;
  }

  tacuda::MACDEXT helper(fastPeriod, slowPeriod, 1, tacuda::MAType::EMA);
  std::vector<float> tmp(3 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(helper, host_input, tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS) {
    return rc;
  }
  std::memcpy(host_output, tmp.data(), size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_macd(const float *host_input, float *host_macd,
                   float *host_signal, float *host_hist, int size,
                   int fastPeriod, int slowPeriod, int signalPeriod,
                   ctMaType_t type, cudaStream_t stream) {
  return api_guard("ct_macd", [&]() -> ctStatus_t {
    if (!host_input || !host_macd || !host_signal || !host_hist) {
      tacuda::detail::set_last_error("ct_macd: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macd: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_macd_ex(host_input, host_macd, host_signal, host_hist, size,
                    fastPeriod, slowPeriod, signalPeriod, type, CT_DTYPE_F32,
                    CT_DTYPE_F32, stream);

  });
}

ctStatus_t ct_macd_ex(const void *host_input, void *host_macd, void *host_signal,
                      void *host_hist, int size, int fastPeriod, int slowPeriod,
                      int signalPeriod, ctMaType_t type, ctDataType_t storageType,
                      ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_macd_ex", [&]() -> ctStatus_t {
    if (!host_input || !host_macd || !host_signal || !host_hist) {
      tacuda::detail::set_last_error("ct_macd_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macd_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_macd || !host_signal || !host_hist || size <= 0) {
    return CT_STATUS_INVALID_ARGUMENT;
  }
  if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
    return CT_STATUS_UNSUPPORTED;
  }

  void* d_in = ct_device_alloc_ex(size, storageType);
  void* d_macd = ct_device_alloc_ex(size, storageType);
  void* d_signal = ct_device_alloc_ex(size, storageType);
  void* d_hist = ct_device_alloc_ex(size, storageType);
  if (!d_in || !d_macd || !d_signal || !d_hist) {
    if (d_in) ct_device_free_ex(d_in);
    if (d_macd) ct_device_free_ex(d_macd);
    if (d_signal) ct_device_free_ex(d_signal);
    if (d_hist) ct_device_free_ex(d_hist);
    return CT_STATUS_ALLOC_FAILED;
  }

  ctStatus_t st = ct_device_upload_ex(d_in, host_input, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    st = ct_macd_ex_d(d_in, d_macd, d_signal, d_hist, size, fastPeriod, slowPeriod,
                      signalPeriod, static_cast<int>(type), storageType, computeType, stream);
  }
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_macd, d_macd, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_signal, d_signal, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_hist, d_hist, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) st = CT_STATUS_DEVICE_ERROR;
  }

  ct_device_free_ex(d_in);
  ct_device_free_ex(d_macd);
  ct_device_free_ex(d_signal);
  ct_device_free_ex(d_hist);
  return st;

  });
}

ctStatus_t ct_macdext(const float *host_input, float *host_macd,
                      float *host_signal, float *host_hist, int size,
                      int fastPeriod, ctMaType_t fastType, int slowPeriod,
                      ctMaType_t slowType, int signalPeriod,
                      ctMaType_t signalType, cudaStream_t stream) {
  return api_guard("ct_macdext", [&]() -> ctStatus_t {
    if (!host_input || !host_macd || !host_signal || !host_hist) {
      tacuda::detail::set_last_error("ct_macdext: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macdext: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_macd || !host_signal || !host_hist || size <= 0) {
    tacuda::detail::set_last_error("MACDEXT: invalid arguments");
    return CT_STATUS_INVALID_ARGUMENT;
  }
  tacuda::MACDEXT macd(fastPeriod, static_cast<tacuda::MAType>(fastType),
                       slowPeriod, static_cast<tacuda::MAType>(slowType),
                       signalPeriod, static_cast<tacuda::MAType>(signalType));
  std::vector<float> tmp(3 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(macd, host_input, tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_macd, tmp.data(), size * sizeof(float));
  std::memcpy(host_signal, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_hist, tmp.data() + 2 * static_cast<size_t>(size), size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_macdfix(const float *host_input, float *host_macd,
                      float *host_signal, float *host_hist, int size,
                      int signalPeriod, cudaStream_t stream) {
  return api_guard("ct_macdfix", [&]() -> ctStatus_t {
    if (!host_input || !host_macd || !host_signal || !host_hist) {
      tacuda::detail::set_last_error("ct_macdfix: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macdfix: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MACDFIX macd(signalPeriod);
  std::vector<float> tmp(3 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(macd, host_input, tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_macd, tmp.data(), size * sizeof(float));
  std::memcpy(host_signal, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_hist, tmp.data() + 2 * static_cast<size_t>(size), size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_mama(const float *host_input, float *host_mama, float *host_fama,
                   int size, float fastLimit, float slowLimit, cudaStream_t stream) {
  return api_guard("ct_mama", [&]() -> ctStatus_t {
    if (!host_input || !host_mama || !host_fama) {
      tacuda::detail::set_last_error("ct_mama: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_mama: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAMA ma(fastLimit, slowLimit);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(ma, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_mama, tmp.data(), size * sizeof(float));
  std::memcpy(host_fama, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_apo(const float *host_input, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_apo", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_apo: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_apo: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::APO apo(fastPeriod, slowPeriod);
  return run_indicator(apo, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_ppo(const float *host_input, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_ppo", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ppo: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ppo: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PPO ppo(fastPeriod, slowPeriod);
  return run_indicator(ppo, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_apo_ma(const float *host_input, float *host_output, int size,
                     int fastPeriod, int slowPeriod, ctMaType_t type,
                     cudaStream_t stream) {
  return api_guard("ct_apo_ma", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_apo_ma: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_apo_ma: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (type < CT_MA_SMA || type > CT_MA_T3) {
      tacuda::detail::set_last_error("ct_apo_ma: invalid MA type");
      return CT_STATUS_INVALID_ARGUMENT;
    }

    tacuda::APO apo(fastPeriod, slowPeriod, static_cast<tacuda::MAType>(type));
    return run_indicator(apo, host_input, host_output, size, 1, stream);
  });
}

ctStatus_t ct_ppo_ma(const float *host_input, float *host_output, int size,
                     int fastPeriod, int slowPeriod, ctMaType_t type,
                     cudaStream_t stream) {
  return api_guard("ct_ppo_ma", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ppo_ma: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ppo_ma: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (type < CT_MA_SMA || type > CT_MA_T3) {
      tacuda::detail::set_last_error("ct_ppo_ma: invalid MA type");
      return CT_STATUS_INVALID_ARGUMENT;
    }

    tacuda::PPO ppo(fastPeriod, slowPeriod, static_cast<tacuda::MAType>(type));
    return run_indicator(ppo, host_input, host_output, size, 1, stream);
  });
}

ctStatus_t ct_pvo(const float *host_volume, float *host_output, int size,
                  int fastPeriod, int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_pvo", [&]() -> ctStatus_t {
    if (!host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_pvo: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_pvo: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PVO pvo(fastPeriod, slowPeriod);
  return run_indicator(pvo, host_volume, host_output, size, 1, stream);

  });
}

ctStatus_t ct_sma_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_sma_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_sma_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sma_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::SMA sma(period);
  return run_indicator_device(sma, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_ma_device(const float *device_input, float *device_output,
                        int size, int period, ctMaType_t type,
                        cudaStream_t stream) {
  return api_guard("ct_ma_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_ma_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ma_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MA ma(period, static_cast<tacuda::MAType>(type));
  return run_indicator_device(ma, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_wma_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_wma_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_wma_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_wma_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::WMA wma(period);
  return run_indicator_device(wma, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_momentum_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  return api_guard("ct_momentum_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_momentum_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_momentum_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Momentum mom(period);
  return run_indicator_device(mom, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_change_device(const float *device_input, float *device_output,
                            int size, int period, cudaStream_t stream) {
  return api_guard("ct_change_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_change_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_change_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Change ch(period);
  return run_indicator_device(ch, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_roc_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_roc_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_roc_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_roc_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROC roc(period);
  return run_indicator_device(roc, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_rocp_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  return api_guard("ct_rocp_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_rocp_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocp_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCP rocp(period);
  return run_indicator_device(rocp, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_rocr_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  return api_guard("ct_rocr_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_rocr_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocr_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCR rocr(period);
  return run_indicator_device(rocr, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_rocr100_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  return api_guard("ct_rocr100_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_rocr100_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rocr100_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ROCR100 rocr100(period);
  return run_indicator_device(rocr100, device_input, device_output, size,
                              stream);

  });
}

ctStatus_t ct_ema_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_ema_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_ema_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ema_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::EMA ema(period);
  return run_indicator_device(ema, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_dema_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  return api_guard("ct_dema_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_dema_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_dema_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DEMA dema(period);
  return run_indicator_device(dema, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_tema_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  return api_guard("ct_tema_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_tema_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_tema_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TEMA tema(period);
  return run_indicator_device(tema, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_t3_device(const float *device_input, float *device_output,
                        int size, int period, float vFactor,
                        cudaStream_t stream) {
  return api_guard("ct_t3_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_t3_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_t3_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::T3 t3(period, vFactor);
  return run_indicator_device(t3, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_trima_device(const float *device_input, float *device_output,
                           int size, int period, cudaStream_t stream) {
  return api_guard("ct_trima_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_trima_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_trima_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TRIMA trima(period);
  return run_indicator_device(trima, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_trix_device(const float *device_input, float *device_output,
                          int size, int period, cudaStream_t stream) {
  return api_guard("ct_trix_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_trix_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_trix_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TRIX trix(period);
  return run_indicator_device(trix, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_max_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_max_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_max_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_max_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAX mx(period);
  return run_indicator_device(mx, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_min_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_min_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_min_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_min_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MIN mn(period);
  return run_indicator_device(mn, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_maxindex_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  return api_guard("ct_maxindex_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_maxindex_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_maxindex_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MAXINDEX mi(period);
  return run_indicator_device(mi, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_minindex_device(const float *device_input,
                              float *device_output, int size, int period,
                              cudaStream_t stream) {
  return api_guard("ct_minindex_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_minindex_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minindex_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MININDEX mi(period);
  return run_indicator_device(mi, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_stddev_device(const float *device_input, float *device_output,
                            int size, int period, cudaStream_t stream) {
  return api_guard("ct_stddev_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_stddev_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_stddev_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StdDev sd(period);
  return run_indicator_device(sd, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_var_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_var_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_var_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_var_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::VAR vr(period);
  return run_indicator_device(vr, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_sum_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_sum_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_sum_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sum_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::SUM sum(period);
  return run_indicator_device(sum, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_rsi_device(const float *device_input, float *device_output,
                         int size, int period, cudaStream_t stream) {
  return api_guard("ct_rsi_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_rsi_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_rsi_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::RSI rsi(period);
  return run_indicator_device(rsi, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_kama_device(const float *device_input, float *device_output,
                          int size, int period, int fastPeriod, int slowPeriod,
                          cudaStream_t stream) {
  return api_guard("ct_kama_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_kama_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_kama_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::KAMA kama(period, fastPeriod, slowPeriod);
  return run_indicator_device(kama, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_macd_line_device(const float *device_input,
                               float *device_output, int size, int fastPeriod,
                               int slowPeriod, cudaStream_t stream) {
  return api_guard("ct_macd_line_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_macd_line_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_macd_line_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (fastPeriod <= 0 || slowPeriod <= 0 || fastPeriod >= slowPeriod) {
    return CT_STATUS_KERNEL_FAILED;
  }
  if (size <= 0) {
    return CT_STATUS_SUCCESS;
  }

  tacuda::MACDEXT helper(fastPeriod, slowPeriod, 1, tacuda::MAType::EMA);

  DeviceBuffer d_tmp{nullptr, PooledDeleter{stream}};
  float *raw = nullptr;
  try {
    raw = static_cast<float *>(DeviceBufferPool::instance().acquire(
        static_cast<size_t>(3) * size * sizeof(float)));
  } catch (...) {
    return CT_STATUS_ALLOC_FAILED;
  }
  d_tmp.reset(raw);

  try {
    helper.calculate(device_input, d_tmp.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  cudaError_t err = cudaMemcpyAsync(device_output, d_tmp.get(),
                                    size * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_apo_device(const float *device_input, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  return api_guard("ct_apo_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_apo_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_apo_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::APO apo(fastPeriod, slowPeriod);
  return run_indicator_device(apo, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_ppo_device(const float *device_input, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  return api_guard("ct_ppo_device", [&]() -> ctStatus_t {
    if (!device_input || !device_output) {
      tacuda::detail::set_last_error("ct_ppo_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ppo_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PPO ppo(fastPeriod, slowPeriod);
  return run_indicator_device(ppo, device_input, device_output, size, stream);

  });
}

ctStatus_t ct_pvo_device(const float *device_volume, float *device_output,
                         int size, int fastPeriod, int slowPeriod,
                         cudaStream_t stream) {
  return api_guard("ct_pvo_device", [&]() -> ctStatus_t {
    if (!device_volume || !device_output) {
      tacuda::detail::set_last_error("ct_pvo_device: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_pvo_device: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PVO pvo(fastPeriod, slowPeriod);
  return run_indicator_device(pvo, device_volume, device_output, size, stream);

  });
}

ctStatus_t ct_accbands(const float *host_high, const float *host_low,
                       const float *host_close, float *host_upper,
                       float *host_middle, float *host_lower, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_accbands", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_upper || !host_middle || !host_lower) {
      tacuda::detail::set_last_error("ct_accbands: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_accbands: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ACCBANDS acc(period);
  std::vector<float> tmp(3 * static_cast<size_t>(size));
  ctStatus_t rc = run_ohlc_indicator(acc, host_high, host_low, host_close,
                                     tmp.data(), size, 3, stream);
  if (rc != CT_STATUS_SUCCESS) {
    return rc;
  }
  std::memcpy(host_upper, tmp.data(), size * sizeof(float));
  std::memcpy(host_middle, tmp.data() + size, size * sizeof(float));
  std::memcpy(host_lower, tmp.data() + 2 * static_cast<size_t>(size), size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_bbands(const float *host_input, float *host_upper,
                     float *host_middle, float *host_lower, int size,
                     int period, float upperMul, float lowerMul, cudaStream_t stream) {
  return api_guard("ct_bbands", [&]() -> ctStatus_t {
    if (!host_input || !host_upper || !host_middle || !host_lower) {
      tacuda::detail::set_last_error("ct_bbands: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_bbands: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_bbands_ex(host_input, host_upper, host_middle, host_lower, size,
                      period, upperMul, lowerMul, CT_DTYPE_F32, CT_DTYPE_F32,
                      stream);

  });
}

ctStatus_t ct_bbands_ex(const void *host_input, void *host_upper, void *host_middle,
                        void *host_lower, int size, int period, float upperMul,
                        float lowerMul, ctDataType_t storageType,
                        ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_bbands_ex", [&]() -> ctStatus_t {
    if (!host_input || !host_upper || !host_middle || !host_lower) {
      tacuda::detail::set_last_error("ct_bbands_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_bbands_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_upper || !host_middle || !host_lower || size <= 0) {
    return CT_STATUS_INVALID_ARGUMENT;
  }
  if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
    return CT_STATUS_UNSUPPORTED;
  }
  void* d_in = ct_device_alloc_ex(size, storageType);
  void* d_u = ct_device_alloc_ex(size, storageType);
  void* d_m = ct_device_alloc_ex(size, storageType);
  void* d_l = ct_device_alloc_ex(size, storageType);
  if (!d_in || !d_u || !d_m || !d_l) {
    if (d_in) ct_device_free_ex(d_in);
    if (d_u) ct_device_free_ex(d_u);
    if (d_m) ct_device_free_ex(d_m);
    if (d_l) ct_device_free_ex(d_l);
    return CT_STATUS_ALLOC_FAILED;
  }
  ctStatus_t st = ct_device_upload_ex(d_in, host_input, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    st = ct_bbands_ex_d(d_in, d_u, d_m, d_l, size, period, upperMul, lowerMul,
                        storageType, computeType, stream);
  }
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_upper, d_u, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_middle, d_m, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_lower, d_l, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) st = CT_STATUS_DEVICE_ERROR;
  }
  ct_device_free_ex(d_in);
  ct_device_free_ex(d_u);
  ct_device_free_ex(d_m);
  ct_device_free_ex(d_l);
  return st;

  });
}

ctStatus_t ct_atr(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, float initial, cudaStream_t stream) {
  return api_guard("ct_atr", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_atr: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_atr: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  return ct_atr_ex(host_high, host_low, host_close, host_output, size, period,
                   initial, CT_DTYPE_F32, CT_DTYPE_F32, stream);

  });
}

ctStatus_t ct_atr_ex(const void *host_high, const void *host_low,
                     const void *host_close, void *host_output, int size,
                     int period, float initial, ctDataType_t storageType,
                     ctDataType_t computeType, cudaStream_t stream) {
  return api_guard("ct_atr_ex", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_atr_ex: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_atr_ex: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_high || !host_low || !host_close || !host_output || size <= 0) {
    return CT_STATUS_INVALID_ARGUMENT;
  }
  if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
    return CT_STATUS_UNSUPPORTED;
  }
  void* d_h = ct_device_alloc_ex(size, storageType);
  void* d_l = ct_device_alloc_ex(size, storageType);
  void* d_c = ct_device_alloc_ex(size, storageType);
  void* d_o = ct_device_alloc_ex(size, storageType);
  if (!d_h || !d_l || !d_c || !d_o) {
    if (d_h) ct_device_free_ex(d_h);
    if (d_l) ct_device_free_ex(d_l);
    if (d_c) ct_device_free_ex(d_c);
    if (d_o) ct_device_free_ex(d_o);
    return CT_STATUS_ALLOC_FAILED;
  }
  ctStatus_t st = ct_device_upload_ex(d_h, host_high, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_upload_ex(d_l, host_low, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) st = ct_device_upload_ex(d_c, host_close, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    st = ct_atr_ex_d(d_h, d_l, d_c, d_o, size, period, initial, storageType,
                     computeType, stream);
  }
  if (st == CT_STATUS_SUCCESS) st = ct_device_download_ex(host_output, d_o, size, storageType, stream);
  if (st == CT_STATUS_SUCCESS) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) st = CT_STATUS_DEVICE_ERROR;
  }
  ct_device_free_ex(d_h);
  ct_device_free_ex(d_l);
  ct_device_free_ex(d_c);
  ct_device_free_ex(d_o);
  return st;

  });
}

ctStatus_t ct_natr(const float *host_high, const float *host_low,
                   const float *host_close, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_natr", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_natr: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_natr: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::NATR natr(period);
  return run_ohlc_indicator(natr, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_trange(const float *host_high, const float *host_low,
                     const float *host_close, float *host_output, int size,
                     cudaStream_t stream) {
  return api_guard("ct_trange", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_trange: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_trange: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TRANGE tr;
  return run_ohlc_indicator(tr, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

ctStatus_t ct_stochastic(const float *host_high, const float *host_low,
                         const float *host_close, float *host_k, float *host_d,
                         int size, int kPeriod, int dPeriod,
                         cudaStream_t stream) {
  return api_guard("ct_stochastic", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_k || !host_d) {
      tacuda::detail::set_last_error("ct_stochastic: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_stochastic: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Stochastic stoch(kPeriod, dPeriod);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_ohlc_indicator(stoch, host_high, host_low, host_close,
                                     tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_stochf(const float *host_high, const float *host_low,
                     const float *host_close, float *host_k, float *host_d,
                     int size, int kPeriod, int dPeriod,
                     cudaStream_t stream) {
  return api_guard("ct_stochf", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_k || !host_d) {
      tacuda::detail::set_last_error("ct_stochf: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_stochf: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StochasticFast stoch(kPeriod, dPeriod);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_ohlc_indicator(stoch, host_high, host_low, host_close,
                                     tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_stochrsi(const float *host_input, float *host_k, float *host_d,
                       int size, int rsiPeriod, int kPeriod, int dPeriod,
                       cudaStream_t stream) {
  return api_guard("ct_stochrsi", [&]() -> ctStatus_t {
    if (!host_input || !host_k || !host_d) {
      tacuda::detail::set_last_error("ct_stochrsi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_stochrsi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StochRSI st(rsiPeriod, kPeriod, dPeriod);
  std::vector<float> tmp(2 * static_cast<size_t>(size));
  ctStatus_t rc = run_indicator(st, host_input, tmp.data(), size, 2, stream);
  if (rc != CT_STATUS_SUCCESS)
    return rc;
  std::memcpy(host_k, tmp.data(), size * sizeof(float));
  std::memcpy(host_d, tmp.data() + size, size * sizeof(float));
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_cci(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_cci", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cci: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cci: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::CCI cci(period);
  return run_ohlc_indicator(cci, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

ctStatus_t ct_adx(const float *host_high, const float *host_low,
                  const float *host_close, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_adx", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_adx: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_adx: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ADX adx(period);
  return run_ohlc_indicator(adx, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

ctStatus_t ct_adxr(const float *host_high, const float *host_low,
                   const float *host_close, float *host_output, int size,
                   int period, cudaStream_t stream) {
  return api_guard("ct_adxr", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_adxr: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_adxr: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ADXR adxr(period);
  return run_ohlc_indicator(adxr, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_plus_dm(const float *host_high, const float *host_low,
                      float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_plus_dm", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_plus_dm: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_plus_dm: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PlusDM pdm(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    pdm.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_minus_dm(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_minus_dm", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_minus_dm: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minus_dm: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MinusDM mdm(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mdm.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_plus_di(const float *host_high, const float *host_low,
                      const float *host_close, float *host_output, int size,
                      int period, cudaStream_t stream) {
  return api_guard("ct_plus_di", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_plus_di: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_plus_di: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PlusDI pdi(period);
  return run_ohlc_indicator(pdi, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

ctStatus_t ct_minus_di(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_minus_di", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_minus_di: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_minus_di: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MinusDI mdi(period);
  return run_ohlc_indicator(mdi, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

// SAR/SAREXT are strictly sequential recurrences: a GPU kernel can only use a
// single thread, which made the host API ~70x slower than TA-Lib on 1M bars
// (kernel time + PCIe transfers). The host API therefore runs the shared
// TA-Lib port directly on the CPU (hybrid approach, see SarTaLib.h); the
// sequential kernels remain available through the tacuda::SAR/SAREXT classes
// for pipelines whose data already lives on the GPU.
ctStatus_t ct_sar(const float *host_high, const float *host_low,
                  float *host_output, int size, float step,
                  float maxAcceleration, cudaStream_t stream) {
  return api_guard("ct_sar", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_sar: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sar: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  (void)stream;
  tacuda::detail::clear_last_error();
  if (!host_high || !host_low || !host_output || size <= 0) {
    tacuda::detail::set_last_error("SAR: invalid arguments");
    return CT_STATUS_INVALID_ARGUMENT;
  }
  tacuda::detail::sar_talib_run(host_high, host_low, size,
                                static_cast<double>(step),
                                static_cast<double>(maxAcceleration),
                                host_output);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_sarext(const float *host_high, const float *host_low,
                     float *host_output, int size, float startValue,
                     float offsetOnReverse, float accInitLong, float accLong,
                     float accMaxLong, float accInitShort, float accShort,
                     float accMaxShort, cudaStream_t stream) {
  return api_guard("ct_sarext", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_sarext: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_sarext: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  (void)stream;
  tacuda::detail::clear_last_error();
  if (!host_high || !host_low || !host_output || size <= 0) {
    tacuda::detail::set_last_error("SAREXT: invalid arguments");
    return CT_STATUS_INVALID_ARGUMENT;
  }
  tacuda::detail::sarext_talib_run(
      host_high, host_low, size, static_cast<double>(startValue),
      static_cast<double>(offsetOnReverse), static_cast<double>(accInitLong),
      static_cast<double>(accLong), static_cast<double>(accMaxLong),
      static_cast<double>(accInitShort), static_cast<double>(accShort),
      static_cast<double>(accMaxShort), host_output);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_aroon(const float *host_high, const float *host_low,
                    float *host_up, float *host_down, float *host_osc, int size,
                    int upPeriod, int downPeriod, cudaStream_t stream) {
  return api_guard("ct_aroon", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_up || !host_down || !host_osc) {
      tacuda::detail::set_last_error("ct_aroon: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_aroon: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Aroon aroon(upPeriod, downPeriod);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size) * 3)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    aroon.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  std::vector<float> tmpHost(3 * size);
  err = cudaMemcpyAsync(tmpHost.data(), d_out.get(), 3 * size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  std::memcpy(host_up, tmpHost.data(), size * sizeof(float));
  std::memcpy(host_down, tmpHost.data() + size, size * sizeof(float));
  std::memcpy(host_osc, tmpHost.data() + 2 * size, size * sizeof(float));

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_aroonosc(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_aroonosc", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_aroonosc: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_aroonosc: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AroonOscillator ind(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ind.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_adosc(const float *host_high, const float *host_low,
                    const float *host_close, const float *host_volume,
                    float *host_output, int size, int shortPeriod,
                    int longPeriod, cudaStream_t stream) {
  return api_guard("ct_adosc", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_adosc: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_adosc: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ADOSC adosc(shortPeriod, longPeriod);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_vol{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    adosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                    d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ad(const float *host_high, const float *host_low,
                 const float *host_close, const float *host_volume,
                 float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_ad", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_ad: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ad: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AD ad;
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_vol{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ad.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                 d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_avgprice(const float *host_open, const float *host_high,
                       const float *host_low, const float *host_close,
                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_avgprice", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_avgprice: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_avgprice: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AvgPrice ap;
  DeviceBuffer d_open{nullptr, PooledDeleter{stream}}, d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}},
      d_close{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_open, elemCount) || !acquire_buffer(d_high, elemCount) ||
      !acquire_buffer(d_low, elemCount) || !acquire_buffer(d_close, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_open.get(), host_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ap.calculate(d_open.get(), d_high.get(), d_low.get(), d_close.get(),
                 d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_medprice(const float *host_high, const float *host_low,
                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_medprice", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_medprice: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_medprice: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MedPrice mp;
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mp.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_typprice(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_typprice", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_typprice: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_typprice: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TypPrice tp;
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    tp.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                 stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_wclprice(const float *host_high, const float *host_low,
                       const float *host_close, float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_wclprice", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_wclprice: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_wclprice: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::WclPrice wc;
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    wc.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(), size,
                 stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_willr(const float *host_high, const float *host_low,
                    const float *host_close, float *host_output, int size,
                    int period, cudaStream_t stream) {
  return api_guard("ct_willr", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_willr: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_willr: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::WILLR willr(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    willr.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                    size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_midpoint(const float *host_input, float *host_output, int size,
                       int period, cudaStream_t stream) {
  return api_guard("ct_midpoint", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_midpoint: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_midpoint: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MIDPOINT mp(period);
  return run_indicator(mp, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_midprice(const float *host_high, const float *host_low,
                       float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_midprice", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_output) {
      tacuda::detail::set_last_error("ct_midprice: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_midprice: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MIDPRICE mp(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_high, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_low, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mp.calculate(d_high.get(), d_low.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ultosc(const float *host_high, const float *host_low,
                     const float *host_close, float *host_output, int size,
                     int shortPeriod, int mediumPeriod, int longPeriod, cudaStream_t stream) {
  return api_guard("ct_ultosc", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_ultosc: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ultosc: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ULTOSC ultosc(shortPeriod, mediumPeriod, longPeriod);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    ultosc.calculate(d_high.get(), d_low.get(), d_close.get(), d_out.get(),
                     size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_mfi(const float *host_high, const float *host_low,
                  const float *host_close, const float *host_volume,
                  float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_mfi", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_mfi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_mfi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MFI mfi(period);
  DeviceBuffer d_high{nullptr, PooledDeleter{stream}}, d_low{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}},
      d_vol{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  const size_t elemCount = static_cast<size_t>(size);

  if (!acquire_buffer(d_high, elemCount) || !acquire_buffer(d_low, elemCount) ||
      !acquire_buffer(d_close, elemCount) || !acquire_buffer(d_vol, elemCount) ||
      !acquire_buffer(d_out, elemCount)) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_high.get(), host_high, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_low.get(), host_low, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_vol.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    mfi.calculate(d_high.get(), d_low.get(), d_close.get(), d_vol.get(),
                  d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_imi(const float *host_open, const float *host_close,
                  float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_imi", [&]() -> ctStatus_t {
    if (!host_open || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_imi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_imi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::IMI imi(period);
  DeviceBuffer d_open{nullptr, PooledDeleter{stream}}, d_close{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_open, static_cast<size_t>(size)) ||
      !acquire_buffer(d_close, static_cast<size_t>(size)) ||
      !acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_open.get(), host_open, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_close.get(), host_close, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    imi.calculate(d_open.get(), d_close.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_obv(const float *host_price, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_obv", [&]() -> ctStatus_t {
    if (!host_price || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_obv: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_obv: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::OBV obv;
  DeviceBuffer d_price{nullptr, PooledDeleter{stream}}, d_volume{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_price, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_volume, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_price.get(), host_price, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_volume.get(), host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    obv.calculate(d_price.get(), d_volume.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_nvi(const float *host_close, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_nvi", [&]() -> ctStatus_t {
    if (!host_close || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_nvi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_nvi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::NVI nvi;
  
  DeviceBuffer d_in{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  if (!acquire_buffer(d_in, 2 * size) || !acquire_buffer(d_out, size)) {
    return CT_STATUS_ALLOC_FAILED;
  }
  
  // Copy close to first half
  cudaError_t err = cudaMemcpyAsync(d_in.get(), host_close, size * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  // Copy volume to second half
  err = cudaMemcpyAsync(d_in.get() + size, host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  try {
    nvi.calculate(d_in.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }
  
  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_pvi(const float *host_close, const float *host_volume,
                  float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_pvi", [&]() -> ctStatus_t {
    if (!host_close || !host_volume || !host_output) {
      tacuda::detail::set_last_error("ct_pvi: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_pvi: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::PVI pvi;
  
  DeviceBuffer d_in{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};
  if (!acquire_buffer(d_in, 2 * size) || !acquire_buffer(d_out, size)) {
    return CT_STATUS_ALLOC_FAILED;
  }
  
  // Copy close to first half
  cudaError_t err = cudaMemcpyAsync(d_in.get(), host_close, size * sizeof(float),
                                    cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  // Copy volume to second half
  err = cudaMemcpyAsync(d_in.get() + size, host_volume, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  try {
    pvi.calculate(d_in.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }
  
  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_beta(const float *host_x, const float *host_y, float *host_output,
                   int size, int period, cudaStream_t stream) {
  return api_guard("ct_beta", [&]() -> ctStatus_t {
    if (!host_x || !host_y || !host_output) {
      tacuda::detail::set_last_error("ct_beta: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_beta: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Beta beta(period);
  DeviceBuffer d_x{nullptr, PooledDeleter{stream}}, d_y{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_x, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_y, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_x.get(), host_x, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_y.get(), host_y, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    beta.calculate(d_x.get(), d_y.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

// Hilbert Transform indicators - CPU-optimized versions
// These are inherently sequential algorithms, so CPU is faster than GPU
ctStatus_t ct_ht_dcperiod(const float *host_input, float *host_output,
                          int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_dcperiod", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ht_dcperiod: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_dcperiod: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_output || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_dcperiod_cpu(host_input, host_output, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ht_dcphase(const float *host_input, float *host_output,
                         int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_dcphase", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ht_dcphase: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_dcphase: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_output || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_dcphase_cpu(host_input, host_output, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ht_phasor(const float *host_input, float *host_inphase,
                        float *host_quadrature, int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_phasor", [&]() -> ctStatus_t {
    if (!host_input || !host_inphase || !host_quadrature) {
      tacuda::detail::set_last_error("ct_ht_phasor: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_phasor: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_inphase || !host_quadrature || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_phasor_cpu(host_input, host_inphase, host_quadrature, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ht_sine(const float *host_input, float *host_sine,
                      float *host_leadsine, int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_sine", [&]() -> ctStatus_t {
    if (!host_input || !host_sine || !host_leadsine) {
      tacuda::detail::set_last_error("ct_ht_sine: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_sine: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_sine || !host_leadsine || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_sine_cpu(host_input, host_sine, host_leadsine, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ht_trendline(const float *host_input, float *host_output,
                           int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_trendline", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ht_trendline: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_trendline: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_output || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_trendline_cpu(host_input, host_output, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_ht_trendmode(const float *host_input, float *host_output,
                           int size, cudaStream_t /*stream*/) {
  return api_guard("ct_ht_trendmode", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_ht_trendmode: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_ht_trendmode: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  if (!host_input || !host_output || size <= 0)
    return CT_STATUS_ALLOC_FAILED;
  tacuda::hilbert_cpu::ht_trendmode_cpu(host_input, host_output, size);
  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_linearreg(const float *host_input, float *host_output, int size,
                        int period, cudaStream_t stream) {
  return api_guard("ct_linearreg", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_linearreg: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_linearreg: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LINEARREG ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_linearreg_slope(const float *host_input, float *host_output,
                              int size, int period, cudaStream_t stream) {
  return api_guard("ct_linearreg_slope", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_linearreg_slope: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_linearreg_slope: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LINEARREG_SLOPE ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_linearreg_intercept(const float *host_input, float *host_output,
                                  int size, int period, cudaStream_t stream) {
  return api_guard("ct_linearreg_intercept", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_linearreg_intercept: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_linearreg_intercept: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LINEARREG_INTERCEPT ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_linearreg_angle(const float *host_input, float *host_output,
                              int size, int period, cudaStream_t stream) {
  return api_guard("ct_linearreg_angle", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_linearreg_angle: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_linearreg_angle: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LINEARREG_ANGLE ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_tsf(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_tsf", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_tsf: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_tsf: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TSF ind(period);
  return run_indicator(ind, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_bop(const float *host_open, const float *host_high,
                  const float *host_low, const float *host_close,
                  float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_bop", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_bop: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_bop: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::BOP bop;
  return run_ohlc_indicator(bop, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_doji(const float *host_open, const float *host_high,
                       const float *host_low, const float *host_close,
                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_doji", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_doji: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_doji: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Doji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_hammer(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_hammer", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_hammer: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_hammer: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Hammer ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_inverted_hammer(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  return api_guard("ct_cdl_inverted_hammer", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_inverted_hammer: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_inverted_hammer: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::InvertedHammer ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_bullish_engulfing(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_bullish_engulfing", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_bullish_engulfing: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_bullish_engulfing: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::BullishEngulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_bearish_engulfing(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_bearish_engulfing", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_bearish_engulfing: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_bearish_engulfing: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::BearishEngulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_white_soldiers(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_white_soldiers", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_white_soldiers: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_white_soldiers: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ThreeWhiteSoldiers ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_abandoned_baby(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_abandoned_baby", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_abandoned_baby: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_abandoned_baby: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AbandonedBaby ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_advance_block(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_advance_block", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_advance_block: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_advance_block: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::AdvanceBlock ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_belt_hold(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_belt_hold", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_belt_hold: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_belt_hold: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::BeltHold ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_breakaway(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_breakaway", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_breakaway: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_breakaway: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Breakaway ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_two_crows(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_two_crows", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_two_crows: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_two_crows: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TwoCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_black_crows(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_black_crows", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_black_crows: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_black_crows: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ThreeBlackCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_inside(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_inside", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_inside: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_inside: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ThreeInside ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_outside(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_outside", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_outside: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_outside: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::CDL3OUTSIDE ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_line_strike(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_line_strike", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_line_strike: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_line_strike: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ThreeLineStrike ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_three_stars_in_south(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_three_stars_in_south", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_three_stars_in_south: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_three_stars_in_south: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ThreeStarsInSouth ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_closing_marubozu(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  return api_guard("ct_cdl_closing_marubozu", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_closing_marubozu: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_closing_marubozu: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ClosingMarubozu ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_conceal_baby_swallow(const float *host_open,
                                       const float *host_high,
                                       const float *host_low,
                                       const float *host_close,
                                       float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_conceal_baby_swallow", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_conceal_baby_swallow: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_conceal_baby_swallow: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ConcealBabySwallow ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_counterattack(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_counterattack", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_counterattack: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_counterattack: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::CounterAttack ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_dark_cloud_cover(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  return api_guard("ct_cdl_dark_cloud_cover", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_dark_cloud_cover: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_dark_cloud_cover: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DarkCloudCover ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_doji_star(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_doji_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_doji_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_doji_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_dragonfly_doji(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_dragonfly_doji", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_dragonfly_doji: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_dragonfly_doji: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DragonflyDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_engulfing(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_engulfing", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_engulfing: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_engulfing: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Engulfing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_evening_doji_star(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_evening_doji_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_evening_doji_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_evening_doji_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::EveningDojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_evening_star(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_evening_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_evening_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_evening_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::EveningStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_gap_side_side_white(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_gap_side_side_white", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_gap_side_side_white: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_gap_side_side_white: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::GapSideSideWhite ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_gravestone_doji(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  return api_guard("ct_cdl_gravestone_doji", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_gravestone_doji: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_gravestone_doji: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::GravestoneDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_hanging_man(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_hanging_man", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_hanging_man: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_hanging_man: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::HangingMan ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_harami(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_harami", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_harami: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_harami: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Harami ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_harami_cross(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_harami_cross", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_harami_cross: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_harami_cross: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::HaramiCross ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_high_wave(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_high_wave", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_high_wave: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_high_wave: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::HighWave ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_hikkake(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_hikkake", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_hikkake: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_hikkake: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Hikkake ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_hikkake_mod(const float *host_open, const float *host_high,
                              const float *host_low, const float *host_close,
                              float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_hikkake_mod", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_hikkake_mod: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_hikkake_mod: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::HikkakeMod ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_homing_pigeon(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_homing_pigeon", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_homing_pigeon: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_homing_pigeon: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::HomingPigeon ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_identical_three_crows(const float *host_open,
                                        const float *host_high,
                                        const float *host_low,
                                        const float *host_close,
                                        float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_identical_three_crows", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_identical_three_crows: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_identical_three_crows: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::IdenticalThreeCrows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_in_neck(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_in_neck", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_in_neck: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_in_neck: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::InNeck ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_kicking(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_kicking", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_kicking: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_kicking: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Kicking ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_kicking_by_length(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_kicking_by_length", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_kicking_by_length: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_kicking_by_length: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::KickingByLength ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_ladder_bottom(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_ladder_bottom", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_ladder_bottom: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_ladder_bottom: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LadderBottom ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_long_legged_doji(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  return api_guard("ct_cdl_long_legged_doji", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_long_legged_doji: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_long_legged_doji: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LongLeggedDoji ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_long_line(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_long_line", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_long_line: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_long_line: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::LongLine ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_marubozu(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_marubozu", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_marubozu: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_marubozu: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Marubozu ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_matching_low(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_matching_low", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_matching_low: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_matching_low: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MatchingLow ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_mat_hold(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_mat_hold", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_mat_hold: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_mat_hold: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MatHold ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_morning_doji_star(const float *host_open,
                                    const float *host_high,
                                    const float *host_low,
                                    const float *host_close, float *host_output,
                                    int size, cudaStream_t stream) {
  return api_guard("ct_cdl_morning_doji_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_morning_doji_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_morning_doji_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MorningDojiStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_morning_star(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_morning_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_morning_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_morning_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::MorningStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_on_neck(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_on_neck", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_on_neck: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_on_neck: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::OnNeck ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_piercing(const float *host_open, const float *host_high,
                           const float *host_low, const float *host_close,
                           float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_piercing", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_piercing: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_piercing: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Piercing ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_rickshaw_man(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_rickshaw_man", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_rickshaw_man: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_rickshaw_man: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::RickshawMan ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_rise_fall3_methods(const float *host_open,
                                     const float *host_high,
                                     const float *host_low,
                                     const float *host_close,
                                     float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_rise_fall3_methods", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_rise_fall3_methods: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_rise_fall3_methods: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::RiseFall3Methods ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_separating_lines(const float *host_open,
                                   const float *host_high,
                                   const float *host_low,
                                   const float *host_close, float *host_output,
                                   int size, cudaStream_t stream) {
  return api_guard("ct_cdl_separating_lines", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_separating_lines: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_separating_lines: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::SeparatingLines ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_shooting_star(const float *host_open, const float *host_high,
                                const float *host_low, const float *host_close,
                                float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_shooting_star", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_shooting_star: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_shooting_star: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ShootingStar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_short_line(const float *host_open, const float *host_high,
                             const float *host_low, const float *host_close,
                             float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_short_line", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_short_line: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_short_line: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::ShortLine ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_spinning_top(const float *host_open, const float *host_high,
                               const float *host_low, const float *host_close,
                               float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_spinning_top", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_spinning_top: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_spinning_top: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::SpinningTop ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_stalled_pattern(const float *host_open,
                                  const float *host_high, const float *host_low,
                                  const float *host_close, float *host_output,
                                  int size, cudaStream_t stream) {
  return api_guard("ct_cdl_stalled_pattern", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_stalled_pattern: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_stalled_pattern: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StalledPattern ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_stick_sandwich(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_stick_sandwich", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_stick_sandwich: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_stick_sandwich: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::StickSandwich ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_takuri(const float *host_open, const float *host_high,
                         const float *host_low, const float *host_close,
                         float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_takuri", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_takuri: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_takuri: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Takuri ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_tasuki_gap(const float *host_open, const float *host_high,
                             const float *host_low, const float *host_close,
                             float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_tasuki_gap", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_tasuki_gap: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_tasuki_gap: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::TasukiGap ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_thrusting(const float *host_open, const float *host_high,
                            const float *host_low, const float *host_close,
                            float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_thrusting", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_thrusting: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_thrusting: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Thrusting ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_tristar(const float *host_open, const float *host_high,
                          const float *host_low, const float *host_close,
                          float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_tristar", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_tristar: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_tristar: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Tristar ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_unique_3_river(const float *host_open, const float *host_high,
                                 const float *host_low, const float *host_close,
                                 float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_unique_3_river", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_unique_3_river: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_unique_3_river: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Unique3River ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_upside_gap_2_crows(const float *host_open,
                                     const float *host_high,
                                     const float *host_low,
                                     const float *host_close,
                                     float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_upside_gap_2_crows", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_upside_gap_2_crows: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_upside_gap_2_crows: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::UpsideGap2Crows ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cdl_xside_gap_3_methods(const float *host_open,
                                      const float *host_high,
                                      const float *host_low,
                                      const float *host_close,
                                      float *host_output, int size, cudaStream_t stream) {
  return api_guard("ct_cdl_xside_gap_3_methods", [&]() -> ctStatus_t {
    if (!host_open || !host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_cdl_xside_gap_3_methods: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cdl_xside_gap_3_methods: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::XSideGap3Methods ind;
  return run_ohlc_indicator(ind, host_open, host_high, host_low, host_close,
                            host_output, size, 1, stream);

  });
}

ctStatus_t ct_cmo(const float *host_input, float *host_output, int size,
                  int period, cudaStream_t stream) {
  return api_guard("ct_cmo", [&]() -> ctStatus_t {
    if (!host_input || !host_output) {
      tacuda::detail::set_last_error("ct_cmo: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_cmo: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::CMO cmo(period);
  return run_indicator(cmo, host_input, host_output, size, 1, stream);

  });
}

ctStatus_t ct_correl(const float *host_x, const float *host_y,
                     float *host_output, int size, int period, cudaStream_t stream) {
  return api_guard("ct_correl", [&]() -> ctStatus_t {
    if (!host_x || !host_y || !host_output) {
      tacuda::detail::set_last_error("ct_correl: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_correl: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::Correl correl(period);
  DeviceBuffer d_x{nullptr, PooledDeleter{stream}}, d_y{nullptr, PooledDeleter{stream}}, d_out{nullptr, PooledDeleter{stream}};

  if (!acquire_buffer(d_x, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_y, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  if (!acquire_buffer(d_out, static_cast<size_t>(size))) {
    return CT_STATUS_ALLOC_FAILED;
  }

  cudaError_t err = cudaMemcpyAsync(d_x.get(), host_x, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }
  err = cudaMemcpyAsync(d_y.get(), host_y, size * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  try {
    correl.calculate(d_x.get(), d_y.get(), d_out.get(), size, stream);
  } catch (...) {
    return classify_exception();
  }

  err = cudaMemcpyAsync(host_output, d_out.get(), size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CT_STATUS_COPY_FAILED;
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CT_STATUS_KERNEL_FAILED;
  }

  return CT_STATUS_SUCCESS;

  });
}

ctStatus_t ct_dx(const float *host_high, const float *host_low,
                 const float *host_close, float *host_output, int size,
                 int period, cudaStream_t stream) {
  return api_guard("ct_dx", [&]() -> ctStatus_t {
    if (!host_high || !host_low || !host_close || !host_output) {
      tacuda::detail::set_last_error("ct_dx: null pointer argument");
      return CT_STATUS_INVALID_ARGUMENT;
    }
    if (size <= 0) {
      tacuda::detail::set_last_error("ct_dx: non-positive size");
      return CT_STATUS_INVALID_ARGUMENT;
    }

  tacuda::DX dx(period);
  return run_ohlc_indicator(dx, host_high, host_low, host_close, host_output,
                            size, 1, stream);

  });
}

// DeviceBufferPool C API
void ct_device_buffer_pool_cleanup(void) {
  DeviceBufferPool::instance().cleanup();
}

size_t ct_device_buffer_pool_allocation_count(void) {
  return DeviceBufferPool::instance().allocationCount();
}

size_t ct_device_buffer_pool_set_max_cache_bytes(size_t bytes) {
  return DeviceBufferPool::instance().setMaxCacheBytes(bytes);
}

} // extern "C"

constexpr ctStatus_t CT_STATUS_INVALID_PARAM = static_cast<ctStatus_t>(-1);
