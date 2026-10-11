#include "tracker.hpp"

#include <ByteTrack/BYTETracker.h>

Tracker::Tracker(int fps)
    : tracker_(std::make_unique<byte_track::BYTETracker>(fps, 30)) {}

Tracker::~Tracker() = default;

void Tracker::update(std::vector<Detection>& detections) {
  std::vector<byte_track::Object> objects;
  objects.reserve(detections.size());
  for (const auto& det : detections) {
    objects.emplace_back(
        byte_track::Rect<float>(det.bbox.x, det.bbox.y, det.bbox.width,
                                det.bbox.height),
        det.classId, det.confidence);
  }

  // ByteTrack returns smoothed boxes without a link to its input, so give
  // each track to the detection it overlaps most
  for (auto& det : detections) det.trackId = -1;
  for (const auto& track : tracker_->update(objects)) {
    const auto& r = track->getRect();
    cv::Rect2f box(r.x(), r.y(), r.width(), r.height());
    Detection* best = nullptr;
    float bestIou = 0.5f;
    for (auto& det : detections) {
      if (det.trackId != -1) continue;
      float overlap = (det.bbox & box).area();
      float iou = overlap / (det.bbox.area() + box.area() - overlap);
      if (iou > bestIou) {
        bestIou = iou;
        best = &det;
      }
    }
    if (best) best->trackId = static_cast<int>(track->getTrackId());
  }
}
