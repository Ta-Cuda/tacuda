# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
cmake_minimum_required(VERSION 3.22)
project(tacuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(BUILD_TESTING "Enable tests" ON)

# -----------------------------------------------------------------------------
# CUDA architecture selection
# -----------------------------------------------------------------------------
# Do not hardcode a single GPU architecture. Let users override via:
# -DCMAKE_CUDA_ARCHITECTURES="75;86" (CMake-native)
# or the convenience cache var below:
set(TACUDA_CUDA_ARCHITECTURES "" CACHE STRING
    "Semicolon-separated CUDA architectures to compile for (e.g., 60;75;86). If empty, uses CMAKE_CUDA_ARCHITECTURES or defaults to a broad set for Compute Capability >= 6.0.")

if (TACUDA_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${TACUDA_CUDA_ARCHITECTURES}" CACHE STRING
      "CUDA architectures" FORCE)
endif()

# If not explicitly overridden, CMake may default to "52" which contradicts the
# project README (Compute Capability >= 6.0). Override the default to a broad
# set covering common Pascal/Turing/Ampere GPUs. Users can narrow this for
# faster builds and smaller binaries.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR
    (NOT TACUDA_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES STREQUAL "52"))
  set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86" CACHE STRING
      "CUDA architectures" FORCE)
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

#Explicitly list indicator sources instead of globbing
set(INDICATOR_SOURCES
    src/indicators/ACCBANDS.cu
    src/indicators/MACD.cu
    src/indicators/MACDEXT.cu
    src/indicators/MACDFIX.cu
    src/indicators/detail/WilderCommon.cu
    src/indicators/Momentum.cu
    src/indicators/MAVP.cu
    src/indicators/ROC.cu
    src/indicators/ROCP.cu
    src/indicators/ROCR.cu
    src/indicators/ROCR100.cu
    src/indicators/SMA.cu
    src/indicators/EMA.cu
    src/indicators/DEMA.cu
    src/indicators/TEMA.cu
    src/indicators/TRIX.cu
    src/indicators/Change.cu
    src/indicators/ClubPack.cu
    src/indicators/StdDev.cu
    src/indicators/AVGDEV.cu
    src/indicators/MA.cu
    src/indicators/MAX.cu
    src/indicators/MAXINDEX.cu
    src/indicators/MAMA.cu
    src/indicators/WMA.cu
    src/indicators/RSI.cu
    src/indicators/KAMA.cu
    src/indicators/BBANDS.cu
    src/indicators/ATR.cu
    src/indicators/Stochastic.cu
    src/indicators/StochasticFused.cu
    src/indicators/CCI.cu
    src/indicators/OBV.cu
    src/indicators/SAR.cu
    src/indicators/ADX.cu
    src/indicators/Aroon.cu
    src/indicators/ADOSC.cu
    src/indicators/MFI.cu
    src/indicators/IMI.cu
    src/indicators/ULTOSC.cu
    src/indicators/AD.cu
    src/indicators/ADXR.cu
    src/indicators/APO.cu
    src/indicators/AroonOscillator.cu
    src/indicators/AvgPrice.cu
    src/indicators/MIN.cu
    src/indicators/MININDEX.cu
    src/indicators/MINMAX.cu
    src/indicators/MINMAXINDEX.cu
    src/indicators/MIDPOINT.cu
    src/indicators/MIDPRICE.cu
    src/indicators/MedPrice.cu
    src/indicators/SAREXT.cu
    src/indicators/StochasticFast.cu
    src/indicators/StochasticFastFused.cu
    src/indicators/Beta.cu
    src/indicators/BOP.cu
    src/indicators/CMO.cu
    src/indicators/Correl.cu
    src/indicators/DX.cu
    src/indicators/PlusDM.cu
    src/indicators/MinusDM.cu
    src/indicators/PlusDI.cu
    src/indicators/MinusDI.cu
    src/indicators/HT_DCPERIOD.cu
    src/indicators/HT_DCPHASE.cu
    src/indicators/HT_PHASOR.cu
    src/indicators/HT_SINE.cu
    src/indicators/HT_TRENDMODE.cu
    src/indicators/HT_TRENDLINE.cu
    src/indicators/LINEARREG.cu
    src/indicators/LINEARREG_ANGLE.cu
    src/indicators/LINEARREG_INTERCEPT.cu
    src/indicators/LINEARREG_SLOPE.cu
    src/indicators/NATR.cu
    src/indicators/NVI.cu
    src/indicators/PPO.cu
    src/indicators/PVI.cu
    src/indicators/PVO.cu
    src/indicators/SUM.cu
    src/indicators/TRANGE.cu
    src/indicators/TRIMA.cu
    src/indicators/T3.cu
    src/indicators/StochRSI.cu
    src/indicators/TSF.cu
    src/indicators/TypPrice.cu
    src/indicators/WclPrice.cu
    src/indicators/WILLR.cu
    src/indicators/VAR.cu
    src/indicators/Doji.cu
    src/indicators/Hammer.cu
    src/indicators/InvertedHammer.cu
    src/indicators/BullishEngulfing.cu
    src/indicators/BearishEngulfing.cu
    src/indicators/ThreeWhiteSoldiers.cu
    src/indicators/AbandonedBaby.cu
    src/indicators/AdvanceBlock.cu
    src/indicators/BeltHold.cu
    src/indicators/Breakaway.cu
    src/indicators/TwoCrows.cu
    src/indicators/ThreeBlackCrows.cu
    src/indicators/ThreeInside.cu
    src/indicators/ThreeLineStrike.cu
    src/indicators/ThreeStarsInSouth.cu
    src/indicators/ClosingMarubozu.cu
    src/indicators/ConcealBabySwallow.cu
    src/indicators/CounterAttack.cu
    src/indicators/CDL3OUTSIDE.cu
    src/indicators/DarkCloudCover.cu
    src/indicators/DojiStar.cu
    src/indicators/DragonflyDoji.cu
    src/indicators/GapSideSideWhite.cu
    src/indicators/GravestoneDoji.cu
    src/indicators/Engulfing.cu
    src/indicators/EveningDojiStar.cu
    src/indicators/EveningStar.cu
    src/indicators/HangingMan.cu
    src/indicators/Harami.cu
    src/indicators/HaramiCross.cu
    src/indicators/HighWave.cu
    src/indicators/Hikkake.cu
    src/indicators/HikkakeMod.cu
    src/indicators/HomingPigeon.cu
    src/indicators/IdenticalThreeCrows.cu
    src/indicators/InNeck.cu
    src/indicators/Kicking.cu
    src/indicators/KickingByLength.cu
    src/indicators/LadderBottom.cu
    src/indicators/LongLeggedDoji.cu
    src/indicators/LongLine.cu
    src/indicators/Marubozu.cu
    src/indicators/MatchingLow.cu
    src/indicators/MatHold.cu
    src/indicators/MorningDojiStar.cu
    src/indicators/MorningStar.cu
    src/indicators/OnNeck.cu
    src/indicators/Piercing.cu
    src/indicators/RickshawMan.cu
    src/indicators/RiseFall3Methods.cu
    src/indicators/SeparatingLines.cu
    src/indicators/ShootingStar.cu
    src/indicators/ShortLine.cu
    src/indicators/SpinningTop.cu
    src/indicators/StalledPattern.cu
    src/indicators/StickSandwich.cu
    src/indicators/Takuri.cu
    src/indicators/TasukiGap.cu
    src/indicators/Thrusting.cu
    src/indicators/Tristar.cu
    src/indicators/Unique3River.cu
    src/indicators/UpsideGap2Crows.cu
    src/indicators/XSideGap3Methods.cu
)

add_library(tacuda SHARED
    ${INDICATOR_SOURCES}
    src/api/api.cpp
    src/api/api_device.cu
    src/api/api_vector_math.cu
    src/api/api_batch.cu
    src/utils/DeviceBufferPool.cu
    src/utils/TacudaConfig.cpp
    src/utils/ErrorState.cpp
    src/context/Context.cu
)
# The public surface is exported explicitly: the C ABI via CTAPI_EXPORT
# (dllexport in tacuda.h / tacuda_device.h) and the C++ Context API via
# TACUDA_API. WINDOWS_EXPORT_ALL_SYMBOLS would additionally leak thousands
# of internal C++/Thrust/CUB symbols into the DLL export table.
# On Linux the same policy is enforced with hidden default visibility:
# only symbols carrying an explicit visibility("default") attribute
# (CTAPI_EXPORT / TACUDA_API) appear in the .so dynamic symbol table.
set_target_properties(tacuda PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(tacuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(tacuda PRIVATE CUDA::cudart)

# Включаем C++17 features для CUDA
target_compile_features(tacuda PRIVATE cxx_std_17)

target_compile_options(tacuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
    >
)

add_executable(example_app src/main.cpp)
target_compile_features(example_app PRIVATE cxx_std_17)
target_include_directories(example_app PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(example_app PRIVATE tacuda CUDA::cudart)

#Tests(simple CTest with a small program using the C API)
if (BUILD_TESTING)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_MakeAvailable(googletest)

  file(GLOB TEST_SOURCES tests/cpp/*.cpp)
  add_executable(test_basic ${TEST_SOURCES})
  target_compile_features(test_basic PRIVATE cxx_std_17)
  target_include_directories(test_basic PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_link_libraries(test_basic PRIVATE tacuda GTest::gtest_main CUDA::cudart)
  include(GoogleTest)
  gtest_discover_tests(test_basic)

  add_test(
    NAME bindings_up_to_date
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bindings/generate_bindings.py --check
  )
endif()

# Strategy optimizer tool (tools/optimizer, target: tacuda-opt)
option(TACUDA_BUILD_OPTIMIZER "Build the tacuda-opt strategy optimizer tool" ON)
if (TACUDA_BUILD_OPTIMIZER)
  add_subdirectory(tools/optimizer)
endif()

# Install and export
install(TARGETS tacuda EXPORT tacudaTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(TARGETS example_app RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT tacudaTargets
        FILE tacudaTargets.cmake
        NAMESPACE tacuda::
        DESTINATION lib/cmake/tacuda)
install(FILES cmake/tacudaConfig.cmake DESTINATION lib/cmake/tacuda)
