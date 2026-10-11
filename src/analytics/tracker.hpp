#pragma once

#include "inference/detection.hpp"

#include <memory>
#include <vector>

namespace byte_track {
class BYTETracker;
}

// ByteTrack over the detections of one stream. Frames must arrive in order.
class Tracker {
public:
  // fps: frame rate of the stream; a lost object keeps its id for one second
  explicit Tracker(int fps = 30);
  ~Tracker();

  // Sets trackId on the detections that belong to a confirmed track
  void update(std::vector<Detection>& detections);

private:
  std::unique_ptr<byte_track::BYTETracker> tracker_;
};
