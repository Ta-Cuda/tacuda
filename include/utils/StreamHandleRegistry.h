#ifndef TACUDA_STREAM_HANDLE_REGISTRY_H
#define TACUDA_STREAM_HANDLE_REGISTRY_H

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace tacuda {
namespace detail {

// Opaque pointer-sized tokens, never addresses to dereference. Odd/even tags
// separate single-series and batch handles. Tokens are not reused: removing
// a live entry releases all of its memory without allowing stale handles to
// alias a later allocation (ABA). Integer/pointer roundtrips are supported on
// our Windows/Linux CUDA targets. Exhaustion fails instead of wrapping.
//
// The mutex protects registry membership for independent handles. Callers
// must still serialize all operations, including destroy, on the SAME handle.
template <typename Handle, typename State, std::uintptr_t Tag>
class StreamHandleRegistry {
  static_assert(Tag == 1 || Tag == 2, "distinct single/batch token tags");
public:
  Handle *insert(std::unique_ptr<State> state) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_ > (std::numeric_limits<std::uintptr_t>::max)() - 2)
      throw std::bad_alloc();
    const auto id = next_;
    entries_.emplace(id, std::move(state));
    next_ += 2;
    return reinterpret_cast<Handle *>(id);
  }

  State *find(const Handle *handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(reinterpret_cast<std::uintptr_t>(handle));
    return it == entries_.end() ? nullptr : it->second.get();
  }

  std::unique_ptr<State> remove(const Handle *handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(reinterpret_cast<std::uintptr_t>(handle));
    if (it == entries_.end()) return nullptr;
    auto state = std::move(it->second);
    entries_.erase(it);
    return state;  // State destruction (possibly CUDA cleanup) is outside lock.
  }

private:
  std::mutex mutex_;
  std::uintptr_t next_ = Tag;
  std::unordered_map<std::uintptr_t, std::unique_ptr<State>> entries_;
};

}  // namespace detail
}  // namespace tacuda

#endif
