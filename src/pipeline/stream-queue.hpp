#pragma once

#include "frame-data.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

// Frames waiting for inference, kept per stream and handed out round-robin.
// A single shared queue would let the streams that happen to push last evict
// the others: with more cameras than the workers can serve, some streams
// starved while others ran at full rate. Here every stream gets the same
// share, and each keeps only its newest `depth` frames.
class StreamQueue {
public:
  StreamQueue(size_t numStreams, size_t depth)
      : slots_(numStreams), depth_(depth) {}

  // Never blocks. If the stream already holds `depth` frames, its oldest is
  // dropped.
  void push(FrameData frame) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& slot = slots_[frame.streamId];
      if (slot.size() >= depth_) slot.pop_front();
      slot.push_back(std::move(frame));
    }
    notEmpty_.notify_one();
  }

  // Blocks until a frame is available or stop() was called. Returns nullopt
  // once stopped and empty.
  std::optional<FrameData> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      // Start after the stream served last, so no stream is favoured
      for (size_t i = 0; i < slots_.size(); i++) {
        auto& slot = slots_[(next_ + i) % slots_.size()];
        if (slot.empty()) continue;
        FrameData frame = std::move(slot.front());
        slot.pop_front();
        next_ = (next_ + i + 1) % slots_.size();
        return frame;
      }
      if (stopped_) return std::nullopt;
      notEmpty_.wait(lock);
    }
  }

  // Frames one stream has waiting
  size_t size(size_t stream) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_[stream].size();
  }

  // Unblocks the consumers; they drain what is queued and then get nullopt
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    notEmpty_.notify_all();
  }

private:
  std::vector<std::deque<FrameData>> slots_;
  size_t depth_;
  size_t next_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable notEmpty_;
  bool stopped_ = false;
};
