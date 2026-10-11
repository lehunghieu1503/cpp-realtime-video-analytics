#include "rules.hpp"

#include <opencv2/imgproc.hpp>

namespace {
// > 0 when p is to the left of the direction a → b
float side(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// True when the segments p1-p2 and a-b cross
bool segmentsCross(const cv::Point2f& p1, const cv::Point2f& p2,
                   const cv::Point2f& a, const cv::Point2f& b) {
  float d1 = side(a, b, p1);
  float d2 = side(a, b, p2);
  float d3 = side(p1, p2, a);
  float d4 = side(p1, p2, b);
  return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}
}  // namespace

Rules::Rules(const std::string& stream, const RulesConfig& config)
    : stream_(stream),
      config_(config),
      lineIn_(config.lines.size(), 0),
      lineOut_(config.lines.size(), 0) {}

Event Rules::makeEvent(const char* type, const std::string& name, int trackId,
                       const TrackState& state, double now) const {
  Event e;
  e.type = type;
  e.stream = stream_;
  e.name = name;
  e.trackId = trackId;
  e.classId = state.classId;
  e.timestamp = now;
  return e;
}

std::vector<Event> Rules::update(const std::vector<Detection>& detections,
                                 double now) {
  std::vector<Event> events;

  for (const auto& det : detections) {
    if (det.trackId < 0) continue;
    cv::Point2f anchor(det.bbox.x + det.bbox.width / 2,
                       det.bbox.y + det.bbox.height);

    auto [it, isNew] = tracks_.try_emplace(det.trackId);
    TrackState& state = it->second;
    if (isNew) {
      state.enteredAt.assign(config_.zones.size(), -1.0);
      state.dwellRaised.assign(config_.zones.size(), false);
    } else {
      for (size_t i = 0; i < config_.lines.size(); i++) {
        const auto& line = config_.lines[i];
        if (!segmentsCross(state.anchor, anchor, line.a, line.b)) continue;
        Event e = makeEvent("line_cross", line.name, det.trackId, state, now);
        bool in = side(line.a, line.b, anchor) > 0;
        e.direction = in ? "in" : "out";
        (in ? lineIn_ : lineOut_)[i]++;
        events.push_back(std::move(e));
      }
    }
    state.anchor = anchor;
    state.classId = det.classId;
    state.lastSeen = now;

    for (size_t i = 0; i < config_.zones.size(); i++) {
      const auto& zone = config_.zones[i];
      bool inside = cv::pointPolygonTest(zone.points, anchor, false) >= 0;
      bool wasInside = state.enteredAt[i] >= 0;
      if (inside && !wasInside) {
        state.enteredAt[i] = now;
        state.dwellRaised[i] = false;
        events.push_back(
            makeEvent("zone_enter", zone.name, det.trackId, state, now));
      } else if (!inside && wasInside) {
        Event e = makeEvent("zone_exit", zone.name, det.trackId, state, now);
        e.dwellSec = now - state.enteredAt[i];
        state.enteredAt[i] = -1.0;
        events.push_back(std::move(e));
      } else if (inside && !state.dwellRaised[i] &&
                 now - state.enteredAt[i] >= config_.dwellSec) {
        Event e = makeEvent("dwell_time", zone.name, det.trackId, state, now);
        e.dwellSec = now - state.enteredAt[i];
        state.dwellRaised[i] = true;
        events.push_back(std::move(e));
      }
    }
  }

  // Forget tracks that disappeared; one still inside a zone leaves it
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    const TrackState& state = it->second;
    if (now - state.lastSeen < config_.lostSec) {
      ++it;
      continue;
    }
    for (size_t i = 0; i < config_.zones.size(); i++) {
      if (state.enteredAt[i] < 0) continue;
      Event e = makeEvent("zone_exit", config_.zones[i].name, it->first,
                          state, now);
      e.dwellSec = state.lastSeen - state.enteredAt[i];
      events.push_back(std::move(e));
    }
    it = tracks_.erase(it);
  }

  return events;
}

std::vector<int> Rules::zoneCounts() const {
  std::vector<int> counts(config_.zones.size(), 0);
  for (const auto& [id, state] : tracks_) {
    for (size_t i = 0; i < counts.size(); i++) {
      if (state.enteredAt[i] >= 0) counts[i]++;
    }
  }
  return counts;
}
