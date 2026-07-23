# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
# tacuda-opt — console strategy optimizer built on the TaCuda C API.
# Design: docs/STRATEGY_OPTIMIZER_PLAN.md, user guide: docs/STRATEGY_OPTIMIZER.md.

cmake_minimum_required(VERSION 3.22)

find_package(Threads REQUIRED)

add_library(tacuda_opt_core STATIC
    src/core/Types.cpp
    src/core/ParamSpace.cpp
    src/core/Fitness.cpp
    src/data/CsvLoader.cpp
    src/data/Synthetic.cpp
    src/backend/Backend.cpp
    src/backend/TacudaBackend.cpp
    src/backend/TalibBackend.cpp
    src/backend/BatchBackend.cpp
    src/backend/TacudaBatchBackend.cpp
    src/engine/IndicatorCache.cpp
    src/engine/Simulator.cpp
    src/engine/Metrics.cpp
    src/engine/MonteCarlo.cpp
    src/strategy/Strategies.cpp
    src/strategy/ExpVol01.cpp
    src/strategy/ExpVol02.cpp
    src/strategy/ExpVol03.cpp
    src/strategy/ExpVol04.cpp
    src/strategy/ExpVol05.cpp
    src/strategy/ExpVol06.cpp
    src/strategy/ExpVol07.cpp
    src/strategy/ExpVol08.cpp
    src/strategy/ExpVol09.cpp
    src/strategy/ExpVol10.cpp
    src/strategy/ExpVol11.cpp
    src/strategy/ExpVol12.cpp
    src/strategy/ExpVol13.cpp
    src/optimize/GridSearch.cpp
    src/optimize/Genetic.cpp
    src/optimize/WalkForward.cpp
    src/portfolio/Portfolio.cpp
    src/report/Report.cpp
    src/cli/Args.cpp
)
target_compile_features(tacuda_opt_core PUBLIC cxx_std_17)
target_include_directories(tacuda_opt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tacuda_opt_core
    PUBLIC tacuda CUDA::cudart Threads::Threads)
if (UNIX)
  target_link_libraries(tacuda_opt_core PUBLIC ${CMAKE_DL_LIBS})
endif()

add_executable(tacuda-opt src/main.cpp)
target_compile_features(tacuda-opt PRIVATE cxx_std_17)
target_link_libraries(tacuda-opt PRIVATE tacuda_opt_core)

if (BUILD_TESTING AND TARGET GTest::gtest_main)
  file(GLOB TACUDA_OPT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(tacuda_opt_tests ${TACUDA_OPT_TEST_SOURCES})
  target_compile_features(tacuda_opt_tests PRIVATE cxx_std_17)
  target_link_libraries(tacuda_opt_tests PRIVATE tacuda_opt_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(tacuda_opt_tests PROPERTIES LABELS "optimizer")
endif()

install(TARGETS tacuda-opt RUNTIME DESTINATION bin)
