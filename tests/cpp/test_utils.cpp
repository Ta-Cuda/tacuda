/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace {

// Cross-platform helper to run Python code and capture output
std::vector<float> run_python_talib(const std::string& python_code, size_t expected_lines) {
    std::vector<float> out(expected_lines, std::numeric_limits<float>::quiet_NaN());
    
    // Create temp file for Python script
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    GetTempFileNameA(temp_path, "tac", 0, temp_file);
    std::string script_path = temp_file;
    // Change extension to .py
    script_path = script_path.substr(0, script_path.length() - 4) + ".py";
    // Rename temp file
    MoveFileA(temp_file, script_path.c_str());
#else
    std::string script_path = "/tmp/tacuda_test_XXXXXX.py";
    int fd = mkstemps(const_cast<char*>(script_path.c_str()), 3);
    if (fd < 0) return out;
    close(fd);
#endif

    // Write Python code to file
    {
        std::ofstream ofs(script_path);
        if (!ofs) return out;
        ofs << "import numpy as np\n";
        ofs << "try:\n";
        ofs << "    import talib\n";
        ofs << "except ImportError:\n";
        ofs << "    import subprocess, sys\n";
        ofs << "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'TA-Lib'])\n";
        ofs << "    import talib\n";
        ofs << python_code;
    }

    // Run Python script
#ifdef _WIN32
    std::string cmd = "python \"" + script_path + "\" 2>nul";
#else
    std::string cmd = "python3 \"" + script_path + "\" 2>/dev/null";
#endif

    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[128];
        for (size_t i = 0; i < out.size() && fgets(buf, sizeof(buf), pipe); ++i) {
            out[i] = std::strtof(buf, nullptr);
        }
        pclose(pipe);
    }

    // Cleanup temp file
#ifdef _WIN32
    DeleteFileA(script_path.c_str());
#else
    unlink(script_path.c_str());
#endif

    return out;
}

// Helper to format input array as Python code
std::string format_input_array(const std::vector<float>& in, const char* var_name = "x") {
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<float>::max_digits10);
    ss << var_name << "=np.array([";
    for (size_t i = 0; i < in.size(); ++i) {
        if (i) ss << ',';
        ss << in[i];
    }
    ss << "],dtype=float)\n";
    return ss.str();
}

// Cross-platform helper for functions returning two arrays (separated by "---")
std::pair<std::vector<float>, std::vector<float>> run_python_talib_pair(
    const std::string& python_code, size_t expected_lines) {
    
    std::vector<float> a(expected_lines, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> b(expected_lines, std::numeric_limits<float>::quiet_NaN());
    
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    GetTempFileNameA(temp_path, "tac", 0, temp_file);
    std::string script_path = temp_file;
    script_path = script_path.substr(0, script_path.length() - 4) + ".py";
    MoveFileA(temp_file, script_path.c_str());
#else
    std::string script_path = "/tmp/tacuda_test_XXXXXX.py";
    int fd = mkstemps(const_cast<char*>(script_path.c_str()), 3);
    if (fd < 0) return {a, b};
    close(fd);
#endif

    {
        std::ofstream ofs(script_path);
        if (!ofs) return {a, b};
        ofs << "import numpy as np\n";
        ofs << "try:\n";
        ofs << "    import talib\n";
        ofs << "except ImportError:\n";
        ofs << "    import subprocess, sys\n";
        ofs << "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'TA-Lib'])\n";
        ofs << "    import talib\n";
        ofs << python_code;
    }

#ifdef _WIN32
    std::string cmd = "python \"" + script_path + "\" 2>nul";
#else
    std::string cmd = "python3 \"" + script_path + "\" 2>/dev/null";
#endif

    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[128];
        size_t idx = 0;
        bool second = false;
        while (fgets(buf, sizeof(buf), pipe)) {
            if (strncmp(buf, "---", 3) == 0) {
                second = true;
                idx = 0;
                continue;
            }
            float v = std::strtof(buf, nullptr);
            if (!second) {
                if (idx < a.size()) a[idx++] = v;
            } else {
                if (idx < b.size()) b[idx++] = v;
            }
        }
        pclose(pipe);
    }

#ifdef _WIN32
    DeleteFileA(script_path.c_str());
#else
    unlink(script_path.c_str());
#endif

    return {a, b};
}

// Cross-platform helper for functions returning three arrays (separated by "@@")
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> run_python_talib_triple(
    const std::string& python_code, size_t expected_lines) {
    
    std::vector<float> v1(expected_lines, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> v2(expected_lines, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> v3(expected_lines, std::numeric_limits<float>::quiet_NaN());
    
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    GetTempFileNameA(temp_path, "tac", 0, temp_file);
    std::string script_path = temp_file;
    script_path = script_path.substr(0, script_path.length() - 4) + ".py";
    MoveFileA(temp_file, script_path.c_str());
#else
    std::string script_path = "/tmp/tacuda_test_XXXXXX.py";
    int fd = mkstemps(const_cast<char*>(script_path.c_str()), 3);
    if (fd < 0) return {v1, v2, v3};
    close(fd);
#endif

    {
        std::ofstream ofs(script_path);
        if (!ofs) return {v1, v2, v3};
        ofs << "import numpy as np\n";
        ofs << "try:\n";
        ofs << "    import talib\n";
        ofs << "except ImportError:\n";
        ofs << "    import subprocess, sys\n";
        ofs << "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'TA-Lib'])\n";
        ofs << "    import talib\n";
        ofs << python_code;
    }

#ifdef _WIN32
    std::string cmd = "python \"" + script_path + "\" 2>nul";
#else
    std::string cmd = "python3 \"" + script_path + "\" 2>/dev/null";
#endif

    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        char buf[256];
        int section = 0;
        size_t idx = 0;
        while (fgets(buf, sizeof(buf), pipe)) {
            if (strncmp(buf, "@@", 2) == 0) {
                section++;
                idx = 0;
                continue;
            }
            float v = std::strtof(buf, nullptr);
            if (section == 0 && idx < v1.size()) v1[idx++] = v;
            else if (section == 1 && idx < v2.size()) v2[idx++] = v;
            else if (section == 2 && idx < v3.size()) v3[idx++] = v;
        }
        pclose(pipe);
    }

#ifdef _WIN32
    DeleteFileA(script_path.c_str());
#else
    unlink(script_path.c_str());
#endif

    return {v1, v2, v3};
}

} // anonymous namespace

// Helper to compare floating point vectors, ignoring NaNs.
void expect_approx_equal(const std::vector<float> &a,
                         const std::vector<float> &b, float eps) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const bool aNan = std::isnan(a[i]);
    const bool bNan = std::isnan(b[i]);
    if (aNan || bNan) {
      EXPECT_EQ(aNan, bNan) << "NaN mismatch at index " << i;
      continue;
    }
    EXPECT_NEAR(a[i], b[i], eps) << "Mismatch at index " << i;
  }
}

std::vector<float> dema_ref(const std::vector<float> &in, int period) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.DEMA(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> tema_ref(const std::vector<float> &in, int period) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.TEMA(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> trix_ref(const std::vector<float> &in, int period) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.TRIX(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> kama_ref(const std::vector<float> &in, int period,
                            int fastPeriod, int slowPeriod) {
  std::ostringstream code;
  code << format_input_array(in);
  // TA-Lib Python wrapper exposes KAMA(timeperiod=...), fast/slow are fixed
  // (2/30) in the reference implementation. Tests pass those defaults.
  (void)fastPeriod;
  (void)slowPeriod;
  code << "res=talib.KAMA(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> beta_ref(const std::vector<float> &x,
                            const std::vector<float> &y, int period) {
  std::ostringstream code;
  code << format_input_array(x, "x");
  code << format_input_array(y, "y");
  code << "res=talib.BETA(x,y,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), x.size());
}

std::vector<float> rsi_ref(const std::vector<float> &in, int period) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.RSI(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> dx_ref(const std::vector<float> &high,
                          const std::vector<float> &low,
                          const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.DX(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> adx_ref(const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.ADX(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> adxr_talib_ref(const std::vector<float> &high,
                                  const std::vector<float> &low,
                                  const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.ADXR(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> natr_ref(const std::vector<float> &high,
                            const std::vector<float> &low,
                            const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.NATR(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> atr_ref(const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.ATR(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::pair<std::vector<float>, std::vector<float>>
stoch_ref(const std::vector<float> &high, const std::vector<float> &low,
          const std::vector<float> &close, int fastkPeriod, int slowPeriod) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "k,d=talib.STOCH(high,low,close,"
          "fastk_period=" << fastkPeriod
       << ",slowk_period=" << slowPeriod << ",slowk_matype=0,"
          "slowd_period=" << slowPeriod << ",slowd_matype=0)\n";
  code << "print('\\n'.join(str(v) for v in k))\n";
  code << "print('---')\n";
  code << "print('\\n'.join(str(v) for v in d))\n";
  return run_python_talib_pair(code.str(), high.size());
}

std::vector<float> t3_ref(const std::vector<float> &in, int period, float vFactor) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.T3(x,timeperiod=" << period << ",vfactor=" << vFactor << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
bbands_ref(const std::vector<float> &in, int period, float upperMul,
           float lowerMul) {
  std::ostringstream code;
  code << format_input_array(in);
  // matype=0 => SMA (TA-Lib default)
  code << "u,m,l=talib.BBANDS(x,timeperiod=" << period << ",nbdevup=" << upperMul
       << ",nbdevdn=" << lowerMul << ",matype=0)\n";
  code << "print('\\n'.join(str(v) for v in u))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in m))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in l))\n";
  return run_python_talib_triple(code.str(), in.size());
}

std::pair<std::vector<float>, std::vector<float>>
mama_ref(const std::vector<float> &in, float fastLimit, float slowLimit) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "m,f=talib.MAMA(x,fastlimit=" << fastLimit << ",slowlimit=" << slowLimit
       << ")\n";
  code << "print('\\n'.join(str(v) for v in m))\n";
  code << "print('---')\n";
  code << "print('\\n'.join(str(v) for v in f))\n";
  return run_python_talib_pair(code.str(), in.size());
}

std::vector<float> sar_talib_ref(const std::vector<float> &high,
                                 const std::vector<float> &low, float step,
                                 float maxAcc) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << "res=talib.SAR(high,low,acceleration=" << step << ",maximum=" << maxAcc << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> sarext_talib_ref(const std::vector<float> &high,
                                    const std::vector<float> &low,
                                    float startValue, float offsetOnReverse,
                                    float accInitLong, float accLong,
                                    float accMaxLong, float accInitShort,
                                    float accShort, float accMaxShort) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << "res=talib.SAREXT(high,low,startvalue=" << startValue
       << ",offsetonreverse=" << offsetOnReverse
       << ",accelerationinitlong=" << accInitLong
       << ",accelerationlong=" << accLong
       << ",accelerationmaxlong=" << accMaxLong
       << ",accelerationinitshort=" << accInitShort
       << ",accelerationshort=" << accShort
       << ",accelerationmaxshort=" << accMaxShort << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macdext_talib_ref(const std::vector<float> &in, int fastPeriod, int fastType,
                  int slowPeriod, int slowType, int signalPeriod,
                  int signalType) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "m,s,h=talib.MACDEXT(x,fastperiod=" << fastPeriod
       << ",fastmatype=" << fastType << ",slowperiod=" << slowPeriod
       << ",slowmatype=" << slowType << ",signalperiod=" << signalPeriod
       << ",signalmatype=" << signalType << ")\n";
  code << "print('\\n'.join(str(v) for v in m))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in s))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in h))\n";
  return run_python_talib_triple(code.str(), in.size());
}

std::vector<float> cdl_ref(const std::vector<float> &open,
                           const std::vector<float> &high,
                           const std::vector<float> &low,
                           const std::vector<float> &close,
                           const char *talibFunc,
                           float penetration) {
  std::ostringstream code;
  code << format_input_array(open, "open");
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  // Some TA-Lib Python wrappers expose a few long CDL* names truncated.
  code << "name='" << talibFunc << "'\n";
  code << "alias={'CDLCONCEALBABYSWALLOW':'CDLCONCEALBABYSWALL'}\n";
  code << "name=alias.get(name,name)\n";
  code << "func=getattr(talib,name)\n";
  if (std::isnan(penetration)) {
    // Use TA-Lib's default penetration (varies by pattern).
    code << "res=func(open,high,low,close)\n";
  } else {
    code << "try:\n";
    code << "    res=func(open,high,low,close,penetration=" << penetration << ")\n";
    code << "except TypeError:\n";
    code << "    res=func(open,high,low,close)\n";
  }
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), open.size());
}

std::vector<float> mavp_ref(const std::vector<float> &in,
                            const std::vector<float> &periods, int minPeriod,
                            int maxPeriod, int maType) {
  std::ostringstream code;
  code << format_input_array(in, "x");
  code << format_input_array(periods, "p");
  code << "res=talib.MAVP(x,p,minperiod=" << minPeriod << ",maxperiod=" << maxPeriod
       << ",matype=" << maType << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> ht_dcperiod_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.HT_DCPERIOD(x)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> ht_dcphase_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.HT_DCPHASE(x)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::pair<std::vector<float>, std::vector<float>>
ht_phasor_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "r1,r2=talib.HT_PHASOR(x)\n";
  code << "print('\\n'.join(str(v) for v in r1))\n";
  code << "print('---')\n";
  code << "print('\\n'.join(str(v) for v in r2))\n";
  return run_python_talib_pair(code.str(), in.size());
}

std::pair<std::vector<float>, std::vector<float>>
ht_sine_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "r1,r2=talib.HT_SINE(x)\n";
  code << "print('\\n'.join(str(v) for v in r1))\n";
  code << "print('---')\n";
  code << "print('\\n'.join(str(v) for v in r2))\n";
  return run_python_talib_pair(code.str(), in.size());
}

std::vector<float> ht_trendmode_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.HT_TRENDMODE(x)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> ht_trendline_ref(const std::vector<float> &in) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.HT_TRENDLINE(x)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macdfix_ref(const std::vector<float> &in, int signalPeriod) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "macd,signal,hist=talib.MACDFIX(x,signalperiod=" << signalPeriod << ")\n";
  code << "print('\\n'.join(str(v) for v in macd))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in signal))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in hist))\n";
  return run_python_talib_triple(code.str(), in.size());
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
macd_ref(const std::vector<float> &in, int fastPeriod, int slowPeriod,
         int signalPeriod) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "macd,signal,hist=talib.MACD(x,fastperiod=" << fastPeriod
       << ",slowperiod=" << slowPeriod << ",signalperiod=" << signalPeriod
       << ")\n";
  code << "print('\\n'.join(str(v) for v in macd))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in signal))\n";
  code << "print('@@')\n";
  code << "print('\\n'.join(str(v) for v in hist))\n";
  return run_python_talib_triple(code.str(), in.size());
}

std::vector<float> macd_line_ref(const std::vector<float> &in, int fastPeriod,
                                 int slowPeriod) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.APO(x,fastperiod=" << fastPeriod
       << ",slowperiod=" << slowPeriod << ",matype=0)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> sar_ref(const std::vector<float> &high,
                           const std::vector<float> &low, float step,
                           float maxAcc) {
  size_t n = high.size();
  std::vector<float> out(n);
  float af = step;
  float ep = high[0];
  float sar = low[0];
  bool longPos = true;
  out[0] = sar;
  for (size_t i = 1; i < n; ++i) {
    sar = sar + af * (ep - sar);
    if (longPos) {
      sar = std::min(sar, low[i - 1]);
      if (low[i] < sar) {
        longPos = false;
        sar = ep;
        ep = low[i];
        af = step;
      } else {
        if (high[i] > ep) {
          ep = high[i];
          af = std::min(af + step, maxAcc);
        }
      }
    } else {
      sar = std::max(sar, high[i - 1]);
      if (high[i] > sar) {
        longPos = true;
        sar = ep;
        ep = high[i];
        af = step;
      } else {
        if (low[i] < ep) {
          ep = low[i];
          af = std::min(af + step, maxAcc);
        }
      }
    }
    out[i] = sar;
  }
  return out;
}

std::vector<float> adosc_ref(
    const std::vector<float> &high, const std::vector<float> &low,
    const std::vector<float> &close, const std::vector<float> &volume,
    int shortP, int longP) {
  int N = high.size();
  std::vector<float> ad(N, 0.0f);
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    float clv =
        denom == 0.0f
            ? 0.0f
            : ((close[i] - low[i]) - (high[i] - close[i])) / denom;
    cum += clv * volume[i];
    ad[i] = cum;
  }
  auto ema_at = [](const std::vector<float> &x, int idx, int period) {
    const float k = 2.0f / (period + 1.0f);
    float weight = 1.0f;
    float weightedSum = x[idx];
    float weightSum = 1.0f;
    int steps = std::min(period, idx);
    for (int i = 1; i <= steps; ++i) {
      weight *= (1.0f - k);
      weightedSum += x[idx - i] * weight;
      weightSum += weight;
    }
    return weightedSum / weightSum;
  };
  // TA-Lib ADOSC lookback = EMA lookback of the slowest period (longP - 1).
  for (int i = longP - 1; i < N; ++i) {
    float emaS = ema_at(ad, i, shortP);
    float emaL = ema_at(ad, i, longP);
    out[i] = emaS - emaL;
  }
  return out;
}

std::vector<float> ultosc_ref(
    const std::vector<float> &high, const std::vector<float> &low,
    const std::vector<float> &close, int shortP, int medP, int longP) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int idx = longP; idx < N; ++idx) {
    float bp1 = 0.0f, tr1 = 0.0f;
    float bp2 = 0.0f, tr2 = 0.0f;
    float bp3 = 0.0f, tr3 = 0.0f;
    for (int j = 0; j < longP; ++j) {
      int i = idx - j;
      float prev = (i > 0) ? close[i - 1] : close[i];
      float bp = close[i] - std::fmin(low[i], prev);
      float tr = std::fmax(high[i], prev) - std::fmin(low[i], prev);
      if (j < shortP) {
        bp1 += bp;
        tr1 += tr;
      }
      if (j < medP) {
        bp2 += bp;
        tr2 += tr;
      }
      bp3 += bp;
      tr3 += tr;
    }
    float avg1 = (tr1 == 0.0f) ? 0.0f : bp1 / tr1;
    float avg2 = (tr2 == 0.0f) ? 0.0f : bp2 / tr2;
    float avg3 = (tr3 == 0.0f) ? 0.0f : bp3 / tr3;
    out[idx] = 100.0f * (4.0f * avg1 + 2.0f * avg2 + avg3) / 7.0f;
  }
  return out;
}

std::vector<float> ad_ref(
    const std::vector<float> &high, const std::vector<float> &low,
    const std::vector<float> &close, const std::vector<float> &volume) {
  int N = high.size();
  std::vector<float> out(N, 0.0f);
  float cum = 0.0f;
  for (int i = 0; i < N; ++i) {
    float denom = high[i] - low[i];
    float clv =
        denom == 0.0f
            ? 0.0f
            : ((close[i] - low[i]) - (high[i] - close[i])) / denom;
    cum += clv * volume[i];
    out[i] = cum;
  }
  return out;
}

float ema_at_ref(const std::vector<float> &x, int idx, int period) {
  const float k = 2.0f / (period + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = std::min(period, idx);
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

float ema_at_ref(const std::vector<float> &x, int idx, int period, int start) {
  const float k = 2.0f / (period + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = std::min(period, idx - start);
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

std::vector<float> apo_ref(const std::vector<float> &in, int fastP,
                           int slowP) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.APO(x,fastperiod=" << fastP << ",slowperiod=" << slowP
       << ",matype=0)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> ppo_ref(const std::vector<float> &in, int fastP,
                           int slowP) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=talib.PPO(x,fastperiod=" << fastP << ",slowperiod=" << slowP
       << ",matype=0)\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> pvo_ref(const std::vector<float> &in, int fastP,
                           int slowP) {
  return ppo_ref(in, fastP, slowP);
}

std::vector<float> aroonosc_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                int period) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = period; i < N; ++i) {
    int sinceHigh = 0, sinceLow = 0;
    float maxVal = high[i];
    float minVal = low[i];
    for (int j = 1; j <= period; ++j) {
      float h = high[i - j];
      float l = low[i - j];
      if (h > maxVal) {  // strict: ties pick the newest bar (TA-Lib)
        maxVal = h;
        sinceHigh = j;
      }
      if (l < minVal) {
        minVal = l;
        sinceLow = j;
      }
    }
    float up = 100.0f * (period - sinceHigh) / period;
    float down = 100.0f * (period - sinceLow) / period;
    out[i] = up - down;
  }
  return out;
}

std::vector<float> adxr_ref(const std::vector<float> &high,
                            const std::vector<float> &low,
                            const std::vector<float> &close, int p) {
  int N = high.size();
  std::vector<float> adx(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> dmp(N, 0.0f), dmm(N, 0.0f), tr(N, 0.0f),
      dx(N, 0.0f);
  for (int i = 1; i < N; ++i) {
    float upMove = high[i] - high[i - 1];
    float downMove = low[i - 1] - low[i];
    dmp[i] = (upMove > downMove && upMove > 0.0f) ? upMove : 0.0f;
    dmm[i] = (downMove > upMove && downMove > 0.0f) ? downMove : 0.0f;
    float range1 = high[i] - low[i];
    float range2 = std::fabs(high[i] - close[i - 1]);
    float range3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::max(range1, std::max(range2, range3));
  }
  float dmp_s = 0.0f, dmm_s = 0.0f, tr_s = 0.0f;
  for (int i = 1; i <= p; ++i) {
    dmp_s += dmp[i];
    dmm_s += dmm[i];
    tr_s += tr[i];
  }
  float dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
  float dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
  dx[p] = (dip + dim == 0.0f)
              ? 0.0f
              : 100.0f * std::fabs(dip - dim) / (dip + dim);
  float dx_sum = dx[p];
  for (int i = p + 1; i < N; ++i) {
    dmp_s = dmp_s - dmp_s / p + dmp[i];
    dmm_s = dmm_s - dmm_s / p + dmm[i];
    tr_s = tr_s - tr_s / p + tr[i];
    dip = (tr_s == 0.0f) ? 0.0f : 100.0f * dmp_s / tr_s;
    dim = (tr_s == 0.0f) ? 0.0f : 100.0f * dmm_s / tr_s;
    dx[i] = (dip + dim == 0.0f)
                ? 0.0f
                : 100.0f * std::fabs(dip - dim) / (dip + dim);
    if (i < 2 * p) {
      dx_sum += dx[i];
      if (i == 2 * p - 1)
        adx[i] = dx_sum / p;
    } else {
      adx[i] = ((adx[i - 1] * (p - 1)) + dx[i]) / p;
    }
  }
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = 3 * p - 1; i < N; ++i) {
    out[i] = 0.5f * (adx[i] + adx[i - p]);
  }
  return out;
}

std::vector<float> avgprice_ref(
    const std::vector<float> &open, const std::vector<float> &high,
    const std::vector<float> &low, const std::vector<float> &close) {
  int N = open.size();
  std::vector<float> out(N, 0.0f);
  for (int i = 0; i < N; ++i)
    out[i] = (open[i] + high[i] + low[i] + close[i]) * 0.25f;
  return out;
}

std::vector<float> typprice_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close) {
  int N = high.size();
  std::vector<float> out(N, 0.0f);
  for (int i = 0; i < N; ++i)
    out[i] = (high[i] + low[i] + close[i]) / 3.0f;
  return out;
}

std::vector<float> wclprice_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                const std::vector<float> &close) {
  int N = high.size();
  std::vector<float> out(N, 0.0f);
  for (int i = 0; i < N; ++i)
    out[i] = (high[i] + low[i] + 2.0f * close[i]) * 0.25f;
  return out;
}

static std::vector<float> run_linearreg_python(
    const std::vector<float> &in, int period, const char *func) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "res=getattr(talib,'" << func << "')(x,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), in.size());
}

std::vector<float> linearreg_ref(const std::vector<float> &in,
                                 int period) {
  return run_linearreg_python(in, period, "LINEARREG");
}

std::vector<float> linearreg_slope_ref(const std::vector<float> &in,
                                       int period) {
  return run_linearreg_python(in, period, "LINEARREG_SLOPE");
}

std::vector<float> linearreg_intercept_ref(const std::vector<float> &in,
                                           int period) {
  return run_linearreg_python(in, period, "LINEARREG_INTERCEPT");
}

std::vector<float> linearreg_angle_ref(const std::vector<float> &in,
                                       int period) {
  return run_linearreg_python(in, period, "LINEARREG_ANGLE");
}

std::vector<float> tsf_ref(const std::vector<float> &in, int period) {
  return run_linearreg_python(in, period, "TSF");
}

std::vector<float> plus_dm_ref(const std::vector<float> &high,
                               const std::vector<float> &low,
                               int period) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  float prevHigh = high[0];
  float prevLow = low[0];
  float prevDM = 0.0f;
  for (int i = 1; i < period; ++i) {
    float diffP = high[i] - prevHigh;
    prevHigh = high[i];
    float diffM = prevLow - low[i];
    prevLow = low[i];
    if (diffP > 0.0f && diffP > diffM)
      prevDM += diffP;
  }
  out[period - 1] = prevDM;
  for (int i = period; i < N; ++i) {
    float diffP = high[i] - prevHigh;
    prevHigh = high[i];
    float diffM = prevLow - low[i];
    prevLow = low[i];
    if (diffP > 0.0f && diffP > diffM)
      prevDM = prevDM - prevDM / period + diffP;
    else
      prevDM = prevDM - prevDM / period;
    out[i] = prevDM;
  }
  return out;
}

std::vector<float> minus_dm_ref(const std::vector<float> &high,
                                const std::vector<float> &low,
                                int period) {
  int N = high.size();
  std::vector<float> out(N, std::numeric_limits<float>::quiet_NaN());
  float prevHigh = high[0];
  float prevLow = low[0];
  float prevDM = 0.0f;
  for (int i = 1; i < period; ++i) {
    float diffP = high[i] - prevHigh;
    prevHigh = high[i];
    float diffM = prevLow - low[i];
    prevLow = low[i];
    if (diffM > 0.0f && diffM > diffP)
      prevDM += diffM;
  }
  out[period - 1] = prevDM;
  for (int i = period; i < N; ++i) {
    float diffP = high[i] - prevHigh;
    prevHigh = high[i];
    float diffM = prevLow - low[i];
    prevLow = low[i];
    if (diffM > 0.0f && diffM > diffP)
      prevDM = prevDM - prevDM / period + diffM;
    else
      prevDM = prevDM - prevDM / period;
    out[i] = prevDM;
  }
  return out;
}

std::vector<float> plus_di_ref(
    const std::vector<float> &high, const std::vector<float> &low,
    const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.PLUS_DI(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::vector<float> minus_di_ref(
    const std::vector<float> &high, const std::vector<float> &low,
    const std::vector<float> &close, int period) {
  std::ostringstream code;
  code << format_input_array(high, "high");
  code << format_input_array(low, "low");
  code << format_input_array(close, "close");
  code << "res=talib.MINUS_DI(high,low,close,timeperiod=" << period << ")\n";
  code << "print('\\n'.join(str(v) for v in res))\n";
  return run_python_talib(code.str(), high.size());
}

std::pair<std::vector<float>, std::vector<float>>
stochrsi_ref(const std::vector<float> &in, int rsiPeriod, int kPeriod,
             int dPeriod) {
  std::ostringstream code;
  code << format_input_array(in);
  code << "k,d=talib.STOCHRSI(x,timeperiod=" << rsiPeriod
       << ",fastk_period=" << kPeriod << ",fastd_period=" << dPeriod
       << ",fastd_matype=0)\n";
  code << "print('\\n'.join(str(v) for v in k))\n";
  code << "print('---')\n";
  code << "print('\\n'.join(str(v) for v in d))\n";
  return run_python_talib_pair(code.str(), in.size());
}
