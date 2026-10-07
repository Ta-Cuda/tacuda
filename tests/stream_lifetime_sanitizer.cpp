// CPU-only regression over the real single-series C API. Compile its source
// with AddressSanitizer too; instrumenting just a consumer misses DLL reads.
#include "tacuda_stream.h"
#include <cstdio>
#include <thread>
#include <vector>

#define CHECK(expression) do { if (!(expression)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression); return false; \
} } while (0)

static bool exercise() {
  ctStream_t *retired = nullptr;
  CHECK(ct_sma_stream_create(3, &retired) == CT_STATUS_SUCCESS);
  CHECK(ct_stream_destroy(retired) == CT_STATUS_SUCCESS);
  for (int i = 0; i < 256; ++i) {
    ctStream_t *live = nullptr;
    CHECK(ct_sma_stream_create(3, &live) == CT_STATUS_SUCCESS);
    CHECK(live != retired);
    float in[] = {1, 2, 3, 4}, out[4] = {};
    CHECK(ct_sma_stream_append(live, in, 4, out) == CT_STATUS_SUCCESS);
    CHECK(out[2] == 2 && out[3] == 3);
    int count = -1;
    long long bars = -1;
    CHECK(ct_stream_reset(retired) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(ct_stream_destroy(retired) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(ct_stream_lookback(retired, &count) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(ct_stream_outputs(retired, &count) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(ct_stream_bars(retired, &bars) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(ct_sma_stream_append(retired, in, 4, out) == CT_STATUS_INVALID_ARGUMENT);
    CHECK(count == -1 && bars == -1);
    CHECK(ct_stream_bars(live, &bars) == CT_STATUS_SUCCESS && bars == 4);
    CHECK(ct_stream_destroy(live) == CT_STATUS_SUCCESS);
  }
  CHECK(ct_stream_destroy(nullptr) == CT_STATUS_SUCCESS);
  return true;
}

int main() {
  if (!exercise()) return 1;
  // Registry membership is shared, but these threads use independent handles.
  int results[4] = {};
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i)
    workers.emplace_back([&, i] { results[i] = exercise() ? 1 : 0; });
  for (auto &worker : workers) worker.join();
  for (int result : results) if (!result) return 1;
  std::puts("PASS: live state, retired handles, allocation churn, independent threads");
  return 0;
}
