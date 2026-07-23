# Device API — Zero-Copy GPU Pipelines

The device API works on raw GPU pointers so multi-indicator pipelines pay
for PCIe transfers once, not per call. Declarations live in
`include/tacuda_device.h` (200 `ct_*_d` functions as of 0.7.1); memory
management is part of the C ABI, so no CUDA headers are required.

## Memory management

```c
float*     ct_device_alloc(int count);                 /* float32 buffer   */
void*      ct_device_alloc_ex(int count, ctDataType_t dtype);
void       ct_device_free(float* d_ptr);
void       ct_device_free_ex(void* d_ptr);
ctStatus_t ct_device_upload(float* d_dst, const float* h_src, int count, cudaStream_t stream);
ctStatus_t ct_device_download(float* h_dst, const float* d_src, int count, cudaStream_t stream);
ctStatus_t ct_device_upload_ex(void* d_dst, const void* h_src, int count, ctDataType_t dtype, cudaStream_t stream);
ctStatus_t ct_device_download_ex(void* h_dst, const void* d_src, int count, ctDataType_t dtype, cudaStream_t stream);
ctStatus_t ct_device_sync(cudaStream_t stream);        /* NULL = default stream */
```

Pass `NULL` for `stream` everywhere unless you manage your own
`cudaStream_t`. Upload/download are asynchronous with respect to the given
stream; call `ct_device_sync` (or download, which orders after the kernels
on the same stream) before reading results.

## Indicator variants

- `ct_<name>_d(...)` — same parameters as the host function, but every
  array argument is a device pointer. All 200 single-series indicators are
  covered, candlestick patterns included.
- `ct_<name>_ex_d(...)` — precision-selectable variants (SMA/EMA/RSI/ATR/
  MACD/BBANDS): `storageType` and `computeType` accept `CT_DTYPE_F16 /
  F32 / F64`.
- `ct_<name>_batch_d(...)` — device-resident multi-symbol matrices, see
  [BATCH_API.md](BATCH_API.md).

Error contract: every entry point clears the thread-local error message on
entry and sets one on any non-success status — `ct_get_last_error()` always
describes the *last* call (see [API.md](API.md#conventions)).

## Example: 3-indicator pipeline, 2 transfers

```c
#include "tacuda.h"
#include "tacuda_device.h"

float* d_in  = ct_device_alloc(n);
float* d_sma = ct_device_alloc(n);
float* d_ema = ct_device_alloc(n);
float* d_rsi = ct_device_alloc(n);

ct_device_upload(d_in, h_prices, n, NULL);      /* PCIe crossing #1 */
ct_sma_d(d_in, d_sma, n, 20, NULL);
ct_ema_d(d_in, d_ema, n, 12, NULL);
ct_rsi_d(d_in, d_rsi, n, 14, NULL);
ct_device_download(h_rsi, d_rsi, n, NULL);      /* PCIe crossing #2 */

ct_device_free(d_in);  ct_device_free(d_sma);
ct_device_free(d_ema); ct_device_free(d_rsi);
```

The same pattern is exposed by every binding:

```python
# Python
import tacuda
d_in = tacuda.device_alloc(n)
tacuda.device_upload(prices, d_in, n)
d_out = tacuda.device_alloc(n)
tacuda.sma_d(d_in, d_out, n, 20)
result = tacuda.device_download(d_out, n)
tacuda.device_free(d_in); tacuda.device_free(d_out)
```

```js
// Node.js
const dIn = tacuda.device.alloc(n);
tacuda.device.upload(dIn, prices);
const dOut = tacuda.device.alloc(n);
tacuda.device.sma(dIn, dOut, n, 20);
const result = tacuda.device.download(dOut, n);
tacuda.device.free(dIn); tacuda.device.free(dOut);
```

```csharp
// C#
IntPtr dIn = NativeMethods.ct_device_alloc(n);
NativeMethods.ct_device_upload(dIn, prices, n, IntPtr.Zero);
IntPtr dOut = NativeMethods.ct_device_alloc(n);
NativeMethods.ct_sma_d(dIn, dOut, n, 20, IntPtr.Zero);
NativeMethods.ct_device_download(result, dOut, n, IntPtr.Zero);
NativeMethods.ct_device_free(dIn); NativeMethods.ct_device_free(dOut);
```

## Buffer pool

Internally tacuda recycles device allocations through a pool with
stream-ordered release (a buffer re-enters the pool only after the work
enqueued on its stream has finished). Controls:

```c
uint64_t ct_device_buffer_pool_allocation_count(void); /* cudaMalloc calls  */
uint64_t ct_device_buffer_pool_set_max_cache_bytes(uint64_t bytes);
void     ct_device_buffer_pool_cleanup(void);          /* drop cached memory */
```

A steady-state workload should show a flat allocation count. On shared GPUs
cap the cache (`set_max_cache_bytes`) or call `cleanup()` between phases.

## When to use which API

| Workload | API |
|----------|-----|
| One indicator, occasionally | host `ct_*` |
| Many indicators on the same series | device `ct_*_d` |
| One indicator over many symbols | batch `ct_*_batch` |
| Many indicators over many symbols, data stays on GPU | `ct_*_batch_d` + `ct_*_d` |

Measured effect on an RTX 3090 Ti: the 7-indicator, 1M-bar pipeline runs
15.4x faster than TA-Lib through the device API (see the
[performance guide](PERFORMANCE.md)).
